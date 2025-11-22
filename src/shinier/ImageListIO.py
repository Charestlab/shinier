from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Any, Optional, Tuple, Union, List, Iterator, Literal, ClassVar, Annotated, get_args, Dict
from PIL import Image
import atexit, shutil, tempfile, weakref, os, time, sys, copy
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, BeforeValidator

from shinier.utils import uint8_plus
from shinier.base import ImageListType, InformativeBaseModel
from shinier.color.Converter import rgb2gray, RGB2GRAY_WEIGHTS
from shinier.Options import ACCEPTED_IMAGE_FORMATS
ACCEPTED_IMAGE_FORMATS = [f".{ext.lower()}" for ext in get_args(ACCEPTED_IMAGE_FORMATS)]


# -----------------------------------------------------------------------------
# Temp management utilities (unchanged)
# -----------------------------------------------------------------------------
_TEMP_ROOT: Optional[Path] = None
_LIVE_DIRS: set[Path] = set()


def cleanup_all_temp_dirs(max_age_hours: int = 0) -> None:
    """
    Force-remove all shinier temp roots (/tmp/shinier-<pid>) regardless of ownership.
    Optionally keep very recent ones by setting max_age_hours > 0.

    Args:
        max_age_hours (int): Minimum age (in hours) for folders to be removed.
                             0 means remove everything immediately.
    """
    tmp_root = Path(tempfile.gettempdir())
    now = time.time()
    for p in tmp_root.glob("shinier-*"):
        if not p.is_dir():
            continue
        age_hours = (now - p.stat().st_mtime) / 3600.0
        if max_age_hours == 0 or age_hours >= max_age_hours:
            shutil.rmtree(p, ignore_errors=True)


def _pid_alive(pid: int) -> bool:
    """Check if a process ID is still alive."""
    try:
        if sys.platform.startswith("win"):
            return True
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _sweep_stale_roots(max_age_hours: int = 168) -> None:
    """Remove old /tmp/shinier-<pid> roots from previous or crashed runs."""
    tmp = Path(tempfile.gettempdir())
    now = time.time()
    for p in tmp.glob("shinier-*"):
        if not p.is_dir():
            continue
        try:
            pid = int(p.name.split("-")[-1])
        except ValueError:
            continue
        age_ok = (now - p.stat().st_mtime) < (max_age_hours * 3600)
        if (not sys.platform.startswith("win") and not _pid_alive(pid)) or not age_ok:
            shutil.rmtree(p, ignore_errors=True)


def _ensure_temp_root() -> Path:
    """Create a single process-scoped temp root lazily."""
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _sweep_stale_roots()
        _TEMP_ROOT = Path(tempfile.gettempdir()) / f"shinier-{os.getpid()}"
        _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        atexit.register(_cleanup_process_root)
    return _TEMP_ROOT


def _register_temp_dir(p: Path) -> None:
    _LIVE_DIRS.add(p)


def _unregister_temp_dir(p: Path) -> None:
    _LIVE_DIRS.discard(p)


def _cleanup_process_root() -> None:
    """Remove all registered temporary dirs and the process root at exit."""
    for p in list(_LIVE_DIRS):
        shutil.rmtree(p, ignore_errors=True)
    if _TEMP_ROOT and _TEMP_ROOT.exists():
        shutil.rmtree(_TEMP_ROOT, ignore_errors=True)


class ImageListIO(InformativeBaseModel):
    """
    Class to manage a list of images with read and write capabilities.
    Inspired by the skimage.io.ImageCollection class.

    Args:
        input_data (ImageListType):
            File pattern, list of file paths, or list of in-memory NumPy arrays.
        conserve_memory (Optional[bool]): If True (default), uses a temporary directory to store images
            and keeps only one image in memory at a time.
        as_gray (Optional[int]): Images are converted into grayscale then uint8 on load only.
            0 = No conversion
            1 = Equal weighted sum of R,G,B
            2 = Rec. ITU-R 601
            3 = Rec. ITU-R 709
            4 = Rec. ITU-R 2020
        save_dir (Optional[str]): Directory to save final images. Defaults to cwd.

    Notes:
        Validity of the collection's image attributes (i.e. dtype, n_dims, n_channels, reference_size) is only guaranteed
            after first initialization. Then, only reference_size is checked. This provides flexibility to update the
            collection with different data type and different channels (e.g. bool -> float). See more information below.
        __getitem__:
            - Height and width of the item should match reference_size.
            - Item will be automatically casted to float if collection dtype is float
            - n_channels is neither validated nor checked.
        __setitem__:
            - Height and width of the new item should match reference_size.
            - New item dtype is accepted as is and determines collection dtype.
            - n_channels is neither validated nor checked.
    """

    # --- Pydantic config ---
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )

    # --- Static class constants ---
    DEFAULT_GRAY_MODE: ClassVar[str] = "L"
    DEFAULT_COLOR_MODE: ClassVar[str] = "RGB"
    ACCEPTED_EXTENSIONS: ClassVar[List[str]] = ACCEPTED_IMAGE_FORMATS

    # --- User-provided attributes ---
    input_data: ImageListType
    conserve_memory: bool = True
    as_gray: Literal[0, 1, 2, 3, 4] = 0
    save_dir: Optional[Path] = None

    # --- Public runtime attributes ---
    data: List[Optional[np.ndarray]] = Field(default_factory=list)
    src_paths: List[Optional[Path]] = Field(default_factory=list)
    store_paths: List[Optional[Path]] = Field(default_factory=list)
    reference_size: Optional[Tuple[int, int]] = Field(default=None)
    n_channels: Optional[int] = Field(default=None)
    n_dims: Optional[int] = Field(default=None)
    n_images: int = Field(default=0)
    dtype: Optional[np.dtype] = Field(default=None)
    drange: Optional[Tuple[Any, Any]] = Field(default=None)
    has_list_array: bool = Field(default=False)

    # --- Private runtime attributes ---
    _temp_dir: Optional[Path] = PrivateAttr(default=None)
    _finalizer: Any = PrivateAttr(default=None)
    _read_only: bool = PrivateAttr(default=False)

    # ------------------------------------------------------------------
    # Post-init constructor (replaces manual __init__)
    # ------------------------------------------------------------------
    def post_init(self, __context: Any) -> None:
        """Perform initialization and collection setup after Pydantic validation."""
        self.data = []
        self.save_dir = Path(self.save_dir or Path.cwd())
        self._initialize_collection(self.input_data)  # This also runs _initial_validation

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> np.ndarray:
        """ Access an image by index from list or store_paths. Cast to collection dtype if necessary."""
        if idx < -self.n_images or idx >= self.n_images:
            raise IndexError("Index out of range.")
        if idx < 0:
            idx = self.n_images + idx
        if self.data[idx] is None:
            if self.conserve_memory:
                # If conserve memory, keep only one image in memory
                self._reset_data()
                img = self._validate_image_attr(self._load_image(self.store_paths[idx]), attr_names=['reference_size'])

                # Cast to float if necessary
                if img.dtype != self.dtype and np.issubdtype(self.dtype, np.floating):
                    img = img.astype(self.dtype, copy=False)

                # img = self._validate_image(self._load_image(self.store_paths[idx]))

                # # Cast to collection dtype if not compatible
                # if self.dtype is not None and img.dtype != self.dtype:
                #     img = img.astype(self.dtype, copy=False)
                self.data[idx] = img
            else:
                raise ValueError(f"Data at index {idx} is None. This should not happen when conserve_memory is False.")

        return self.data[idx]

    def __setitem__(self, idx: int, new_image: np.ndarray) -> None:
        """Modify an image at a given index. Do not cast to current dtype — modify dataset dtype with new image.dtype"""
        if getattr(self, "_read_only", False):
            raise RuntimeError("This ImageListIO fork is read-only; cannot modify items.")

        if idx < 0 or idx >= self.n_images:
            raise IndexError("Index out of range.")

        # Must be the same height and width
        new_image = self._validate_image_attr(new_image, attr_names=['reference_size'])

        # Update collection dtype, n_channels and n_dims (but not enforced: attributes might thus not be accurate)
        self._update_image_attr(new_image, attr_names=['dtype', 'n_channels', 'n_dims'])

        # new_image = self._validate_image(new_image)
        # new_image = self._to_gray(new_image)
        if self.conserve_memory:
            self._reset_data()
            self._save_image(idx, new_image, save_dir=self._temp_dir)

        self.data[idx] = new_image

    def __len__(self) -> int:
        """ Get the number of images in the collection. """
        return self.n_images

    def __iter__(self) -> Iterator[np.ndarray]:
        """ Iterate over the images in the collection. """
        for idx in range(self.n_images):
            yield self[idx]

    def readonly_copy(self):
        """Produce a read-only copy of the instance."""
        # Use Pydantic's safe model duplication
        new = self.model_copy(deep=True)

        # Reset runtime-only attributes
        new._temp_dir = None
        new._finalizer = None

        # Mark as read-only by wrapping __setitem__
        new._read_only = True

        # Clear pixel arrays for memory efficiency
        new.data = [None] * self.n_images

        # No temp/staging is carried over
        new._staging_dir = None
        new._staging_paths = None
        new._staging_enabled = False

        return new

    def new_copy(self, to_list: bool = False) -> ImageListIO:
        # Construct a new instance with or without list of images.
        # This will run normal model validation and initialization.
        input_data = self.to_list() if to_list else self.input_data

        # Initialize new instance with original input_data
        new_instance = self.__class__(input_data=input_data, conserve_memory=self.conserve_memory, as_gray=self.as_gray)

        # Update new instance's image data
        if not to_list:
            for idx, im in enumerate(self):
                new_instance[idx] = im
        return new_instance

    def copy_with_image_list(self) -> ImageListIO:
        # Construct a new instance. This will run normal model validation and initialization.
        # return self.__class__(input_data=self.to_list(), conserve_memory=False)
        return self.new_copy(to_list=True)

    def to_list(self):
        """Produce a list of numpy arrays, one for each image in the collection."""
        return list(self)

    def _ensure_temp_dir(self):
        """Create this instance's temp dir lazily under the process root."""
        if self._temp_dir is None:
            root = _ensure_temp_root()
            self._temp_dir = Path(tempfile.mkdtemp(dir=root, prefix="imagelist-"))
            _register_temp_dir(self._temp_dir)
            # backstop: remove on GC
            self._finalizer = weakref.finalize(self, self._cleanup_finalizer, self._temp_dir)

    @staticmethod
    def _cleanup_finalizer(path: Path):
        try:
            shutil.rmtree(path, ignore_errors=True)
        finally:
            _unregister_temp_dir(path)

    def _cleanup_temp_dir(self) -> None:
        """Idempotent explicit cleanup."""
        if getattr(self, '_temp_dir', None) is None:
            return
        try:
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        finally:
            _unregister_temp_dir(self._temp_dir)
            self._temp_dir = None
            if self._finalizer:
                self._finalizer()  # mark finalizer as done
                self._finalizer = None

    def _validate_image_attr(self, image: np.ndarray, attr_names: List[str], update_nonexistant: bool = True) -> np.ndarray:
        """ Validate the image and return it. """
        for attr_name in attr_names:
            value = getattr(image, attr_name, None)
            if value is None:
                if update_nonexistant:
                    self._update_image_attr(image, [attr_name])
            else:
                if attr_name == "reference_size":
                    if value != image.shape[:2]:
                        raise ValueError(f"Image size {image.shape[:2]} does not match reference size {value}.")
                elif attr_name == "n_dims":
                    if value != image.ndim:
                        raise ValueError(f"Image ndim {image.ndim} do not match collection n_dims {value}.")
                elif attr_name == "n_channels":
                    n_channels = image.shape[2] if value == 3 else 1
                    if value != n_channels:
                        raise ValueError(f"Image has {n_channels} channels which does not match collection n_channels {value}.")
                elif attr_name == "dtype":
                    if value != image.dtype:
                        raise ValueError(f"Image dtype {image.dtype} does not match collection dtype {value}.")
        return image

    def to_gray(self):
        """Public-facing version of _to_gray() that converts all images to grayscale."""
        for idx, img in enumerate(self.images):
            self.images[idx] = self._to_gray(img)

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            if self.as_gray > 0:
                image = rgb2gray(image, conversion_type=RGB2GRAY_WEIGHTS['int2key'][self.as_gray])
                image = uint8_plus(image)
        return image

    def _is_image(self, file: Path) -> bool:
        """Check if a file is an image."""
        return file.is_file() and len(file.suffix) > 1 and file.suffix.lower() in self.ACCEPTED_EXTENSIONS

    def _initial_validation(self, image: np.ndarray) -> np.ndarray:
        """Update image attribute if non-existent and make sure all images have the same attributes"""
        self._update_image_attr(image, ['reference_size', 'n_dims', 'n_channels', 'dtype'], force_update=False)
        return self._validate_image_attr(image, ['reference_size', 'n_dims', 'n_channels', 'dtype'], update_nonexistant=False)

    def _initialize_collection(self, input_data: ImageListType) -> None:
        """ Initialize the image collection from input data. """

        # Type checks
        if not (isinstance(input_data, (str, Path)) or (isinstance(input_data, list) and all(isinstance(d, (np.ndarray, str, Path)) for d in input_data))):
            raise ValueError("Input must be str|Path (glob) or list of str|Path|np.ndarray")
        if isinstance(input_data, list) and all(isinstance(d, np.ndarray) for d in input_data):
            if not all(
                    np.issubdtype(d.dtype, np.integer) or
                    np.issubdtype(d.dtype, np.floating) or
                    np.issubdtype(d.dtype, np.bool_)
                    for d in input_data):
                raise ValueError("Unsupported dtype in list of images")

        # Initialize collection
        if isinstance(input_data, (str, Path)):
            # Convert to Path if input_data is a string
            input_path = Path(input_data)

            # Handle cases with wildcards
            if "*" in str(input_path):  # Check if it's a glob pattern
                directory = input_path.parent
                pattern = input_path.name
                all_files = sorted(directory.glob(pattern))
            else:
                all_files = [input_path] if input_path.is_file() else sorted(input_path.glob("*"))
            # Filter to include only recognized image files
            self.src_paths = sorted([p for p in all_files if self._is_image(p)])
            if not self.src_paths:
                raise FileNotFoundError(f"No files found matching pattern '{input_data}'")

            # Initial store: if not conserving memory yet, the files themselves are the store
            self.store_paths = list(self.src_paths)
            self.n_images = len(self.store_paths)
        elif isinstance(input_data, list):
            self.n_images = len(input_data)
            if all(isinstance(item, np.ndarray) for item in input_data):
                # No provenance for in-memory arrays
                self.src_paths = [None] * self.n_images
                if self.conserve_memory:
                    # Write temp .npy files; those become the backing store
                    self._ensure_temp_dir()
                    self.store_paths = [self._temp_dir / f'image_{idx}.npy' for idx in range(self.n_images)]
                    for idx, im in enumerate(input_data):
                        self._save_image(idx, self._to_gray(im), save_dir=self._temp_dir)
                    self._reset_data()  # Data will not be stored in self.data when conserve_memory is True
                    self.data[0] = self._initial_validation(self._load_image(self.store_paths[0]))
                else:
                    self.has_list_array = True
                    self.data = [self._to_gray(self._initial_validation(image)) for image in input_data]
                    self._update_drange()

                    if self.store_paths.__len__() == 0:
                        self.store_paths = [None] * self.n_images
            elif all(isinstance(item, (str, Path)) for item in input_data):
                self.src_paths = [Path(item) for item in input_data]
                self.store_paths = list(self.src_paths)
            else:
                raise TypeError("input_data must be a file pattern, list of file paths, or list of NumPy arrays.")
        else:
            raise TypeError("input_data must be a file pattern, list of file paths, or list of NumPy arrays.")

        if len(self.store_paths) != self.n_images:
            self.store_paths = [None] * self.n_images

        if not self.data or all(d is None for d in self.data):
            if self.conserve_memory:
                # Only load the first image to initialize attributes
                self._reset_data()  # Data will not be stored in self.data when conserve_memory is True
                self.data[0] = self._initial_validation(self._load_image(self.store_paths[0]))
            elif not self.data and all([isinstance(fp, (str, Path)) for fp in self.store_paths]):
                # Load all images into self.data --- This could not happen:
                self.data = [self._initial_validation(self._load_image(fpath)) for fpath in self.store_paths]
            elif not all([isinstance(d, np.ndarray) for d in self.data]):
                raise ValueError('Input data should be either a list of np.ndarray or a glob pattern or a list of Path.')

        # Make sure all values have been initialized
        self._initial_validation(self.data[0])

    def _update_image_attr(self, image: np.ndarray, attr_names: List[str], force_update: bool = True) -> None:
        """Update image attributes if force_update is True or attribute is None"""
        for attr_name in attr_names:
            if attr_name == 'n_dims':
                self.n_dims = image.ndim if self.n_dims is None or force_update else self.n_dims
            elif attr_name == 'n_channels':
                n_channels = image.shape[-1] if image.ndim == 3 else 1
                self.n_channels = n_channels if self.n_channels is None or force_update else self.n_channels
            elif attr_name == 'reference_size':
                self.reference_size = image.shape[:2] if self.reference_size is None or force_update else self.reference_size
            elif attr_name == 'dtype':
                self.dtype = image.dtype if self.dtype is None or force_update else self.dtype
                self._update_drange()
            elif attr_name == 'drange':
                self._update_drange()
            elif attr_name == 'n_images':
                self.n_images = len(self.data) if self.n_images is None or force_update else self.n_images

    def _update_drange(self) -> None:
        """Update numeric dynamic range based on current self.dtype."""
        if self.dtype is None:
            self.drange = None
            return
        if np.issubdtype(self.dtype, np.bool_) or self.dtype is bool:
            self.drange = (0, 1)
        elif np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            self.drange = (int(info.min), int(info.max))
        elif np.issubdtype(self.dtype, np.floating):
            # Do not assume a fixed range for floats
            pass
        else:
            self.drange = None

    def _load_image(self, image_path: Path) -> np.ndarray:
        """ Load an image from a file path. """
        image = None
        try:
            if image_path.suffix == ".npy":
                image = np.load(image_path)
                # self.dtype = image.dtype
            else:
                with Image.open(image_path) as pil_image:
                    # Load as RGB and convert to grayscale if required
                    pil_image = pil_image.convert(self.DEFAULT_COLOR_MODE)
                    image = np.array(pil_image)
                    image = self._to_gray(image)

                # self.dtype = image.dtype
            # self._update_drange()
        except IOError as e:
            raise IOError(f"Failed to load image from {image_path}: {e}")

        return image

    def _save_image(self, idx: int, image: np.ndarray, save_dir: Optional[Path] = None) -> None:
        """ Save an image to the temporary directory. """
        if getattr(self, "_read_only", False):
            raise RuntimeError("This ImageListIO fork is read-only; cannot modify items.")

        if save_dir is None and self.conserve_memory:
            self._ensure_temp_dir()
            save_dir = self._temp_dir
        else:
            save_dir = Path(save_dir or self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # save_dir = Path(save_dir or self._temp_dir or self.store_paths[idx].parent or Path.cwd())
        # save_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Choose a base name:
            #   1) if we have a provenance name, keep it
            #   2) else reuse current store name
            #   3) else synthesize
            base_name = (
                (self.src_paths[idx].name if self.src_paths[idx] is not None else None)
                or (self.store_paths[idx].name if idx < len(self.store_paths) else None)
                or f"image_{idx}.npy"
            )
            # base_name = self.store_paths[idx].name
            image_path = save_dir / base_name
            file_format = self.get_file_format(image_path)
            self.store_paths[idx] = image_path  # Update file path
            try:
                if file_format == '.npy':
                    np.save(image_path, image.squeeze())
                else:
                    arr = np.asarray(image).squeeze()
                    pil_image = Image.fromarray(arr)

                    # Choose format-specific, as-lossless-as-possible parameters.
                    save_kwargs = {}
                    if file_format == "JPEG":
                        # JPEG is inherently lossy. These settings minimise loss but cannot make it truly lossless.
                        # If you need strictly histogram-preserving behaviour, avoid JPEG and use PNG/TIFF/NPY instead.
                        save_kwargs.update(
                            {
                                "quality": 100,      # max quality
                                "subsampling": 0,    # 4:4:4 chroma
                                "optimize": False,   # deterministic, no extra heuristics
                            }
                        )
                    elif file_format == "PNG":
                        # PNG is lossless; compression only affects size, not pixel values.
                        pass
                    elif file_format == "TIFF":
                        # Use a lossless TIFF compression scheme.
                        # "raw" = no compression; "tiff_lzw" is also lossless smaller files is required.
                        save_kwargs.update({"compression": "raw"})
                    elif file_format == "BMP":
                        # BMP is uncompressed by design; nothing to add.
                        pass
                    pil_image.save(image_path, format=file_format, **save_kwargs)
            except (IOError, TypeError) as e:
                raise IOError(f"Failed to save image at index {idx} to {image_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Failed to save image at index {idx}: {e}")

    def _reset_data(self) -> None:
        """ Reset data attribute with placeholders. """
        self.data = [None] * self.n_images

    @staticmethod
    def get_file_format(image_path: Path) -> str:
        """ Get the file format based on the file extension. """
        ext = image_path.suffix.lower()
        format_mapping = {
            '.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG',
            '.bmp': 'BMP', '.tiff': 'TIFF', '.tif': 'TIFF', '.npy': '.npy'
        }
        return format_mapping.get(ext, 'PNG')

    def final_save_all(self) -> None:
        """ Save images to save_dir. If needed (self.conserve_memory) loads images and clears up temp files. """
        for idx in range(self.n_images):
            self._save_image(idx, self[idx], save_dir=self.save_dir)

        # Clean up temporary directory
        self._cleanup_temp_dir()

    def close(self) -> None:
        self._cleanup_temp_dir()

    def __del__(self) -> None:
        """ Clean up the temporary directory upon object destruction. """
        self._cleanup_temp_dir()
