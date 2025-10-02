# External package imports
from pathlib import Path
import numpy as np
from typing import Any, Optional, Tuple, Union, NewType, List, Iterator, Callable, Literal
from PIL import Image
import atexit, shutil, tempfile, weakref, os, time, sys
from shinier.utils import rgb2gray, uint8_plus
import copy

# Type definition
ImageListType = Union[str, Path, List[Union[str, Path]], List[np.ndarray]]


# ---- Temp management (tiny, in-file) ---------------------------------
_TEMP_ROOT: Optional[Path] = None  # created lazily: /tmp/shinier-<pid>
_LIVE_DIRS: set[Path] = set()  # per-instance temp dirs to remove at exit


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
    # Best effort; on Windows kill(pid, 0) may raise, so we fall back to age-based cleanup.
    try:
        if sys.platform.startswith("win"):
            return True  # skip PID test on Windows; we'll use age cutoff
        else:
            os.kill(pid, 0)  # does not kill; checks permission/existence
            return True
    except OSError:
        return False


def _sweep_stale_roots(max_age_hours: int = 168) -> None:
    """Remove old /tmp/shinier-<pid> roots from previous/crashed runs."""
    tmp = Path(tempfile.gettempdir())
    now = time.time()
    for p in tmp.glob("shinier-*"):
        if not p.is_dir():
            continue
        try:
            pid = int(p.name.split("-")[-1])
        except ValueError:
            continue
        # Remove if PID not alive (Unix) OR folder is older than cutoff
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
    """On normal interpreter exit, remove all registered dirs and the root."""
    for p in list(_LIVE_DIRS):
        shutil.rmtree(p, ignore_errors=True)
    if _TEMP_ROOT and _TEMP_ROOT.exists():
        shutil.rmtree(_TEMP_ROOT, ignore_errors=True)
# ----------------------------------------------------------------------


class ImageListIO:
    """
    Class to manage a list of images with read and write capabilities.
    Inspired by the skimage.io.ImageCollection class.

    Args:
        input_data (ImageListType):
            File pattern, list of file paths, or list of in-memory NumPy arrays.
        conserve_memory (Optional[bool]): If True (default), uses a temporary directory to store images
            and keeps only one image in memory at a time. If True and input_data is a list of NumPy arrays,
            images are first saved as .npy in a temporary directory, and they are loaded in memory one at a time upon request.
        as_gray (Optional[int]): Images are converted into grayscale then uint8 on load only. Default is no conversion (default = 0).
            0 = No conversion applied
            1 = An equal weighted sum of red, green and blue pixels is applied.
            2 = (legacy mode) Rec.ITU-R 601 is used (see Matlab). Y′ = 0.299 R′ + 0.587 G′ + 0.114 B′
            3 = Rec.ITU-R 709 is used. Y′ = 0.2126 R′ + 0.7152 G′ + 0.0722 B′
            4 = Rec.ITU-R 2020 is used. Y′ = 0.2627 R′ + 0.6780 G′ + 0.0593 B′
        save_dir (Optional[str]): Directory to save final images. Defaults to the
            current working directory if not specified.

    Attributes:
        data: The list of images.
        src_paths: The list of original file paths or identifiers.
        store_paths: The list of current file paths or identifiers.
        reference_size: Reference image size (x, y) for validation.
        n_dims: Number of dimensions of the image
    """

    DEFAULT_GRAY_MODE = 'L'
    DEFAULT_COLOR_MODE = 'RGB'

    def __init__(
        self,
        input_data: ImageListType,
        conserve_memory: bool = True,
        as_gray: Literal[0, 1, 2, 3, 4] = 0,
        save_dir: Optional[str] = None
    ) -> None:
        self.conserve_memory: bool = conserve_memory
        self.as_gray: Literal[0, 1, 2, 3, 4] = int(as_gray) if as_gray is not None else 0
        if self.as_gray not in (0, 1, 2, 3, 4):
            raise ValueError("as_gray must be 0 (no conversion), 1 (equal), 2 (Rec. ITU-R 601), 3 (Rec. ITU-R 709), 4 (Rec. ITU-R 2020)")
        self.save_dir: Path = Path(save_dir or Path.cwd())
        self.data: List[Optional[np.ndarray]] = []
        self.src_paths: List[Optional[Path]] = []  # immutable provenance
        self.store_paths: List[Path] = []
        self.reference_size: Optional[Tuple[int, int]] = None
        self.n_dims: Optional[int] = None
        self.n_images: int = 0
        self._temp_dir: Optional[Path] = None
        self._finalizer = None
        self.dtype: Optional[type] = None  # Initial state when loading images
        self.drange: Optional[tuple] = None
        self.has_list_array: bool = False
        self._read_only: bool = False
        self._initialize_collection(input_data)

    def __getitem__(self, idx: int) -> np.ndarray:
        """ Access an image by index. """
        if idx < -self.n_images or idx >= self.n_images:
            raise IndexError("Index out of range.")
        if idx < 0:
            idx = self.n_images + idx
        if self.data[idx] is None:
            if self.conserve_memory:
                # If conserve memory, keep only one image in memory
                self._reset_data()
                img = self._validate_image(self._load_image(self.store_paths[idx]))
                # Cast to collection dtype if not compatible
                if self.dtype is not None and img.dtype != self.dtype:
                    img = img.astype(self.dtype, copy=False)
                self.data[idx] = img
            else:
                raise ValueError(f"Data at index {idx} is None. This should not happen when conserve_memory is False.")
        return self.data[idx]

    def __setitem__(self, idx: int, new_image: np.ndarray) -> None:
        """ Modify an image at a given index. """
        if getattr(self, "_read_only", False):
            raise RuntimeError("This ImageListIO fork is read-only; cannot modify items.")

        if idx < 0 or idx >= self.n_images:
            raise IndexError("Index out of range.")

        self.dtype = new_image.dtype
        self._update_drange()
        new_image = self._validate_image(new_image)
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
        """Produce a read-only copy of an instance."""
        cls = self.__class__
        new = cls.__new__(cls)

        # --- simple attributes copied verbatim ---
        new.conserve_memory = self.conserve_memory
        new.as_gray = self.as_gray
        new.save_dir = self.save_dir

        # --- path metadata: new lists, same Path objects (lazy, no bytes copied) ---
        new.src_paths = list(self.src_paths)
        new.store_paths = list(self.store_paths)

        # --- detached runtime state ---
        new.n_images = self.n_images
        new.reference_size = self.reference_size
        new.n_dims = self.n_dims
        new.has_list_array = self.has_list_array

        # Keep dtype/drange (or set to None if you prefer re-infer)
        new.dtype = copy.copy(self.dtype)
        new.drange = copy.copy(self.drange)

        # Fresh, empty caches; do NOT copy pixel arrays
        new.data = [None] * self.n_images

        # No temp/staging is carried over
        new._temp_dir = None
        new._finalizer = None
        new._staging_dir = None
        new._staging_paths = None
        new._staging_enabled = False

        # Mark as read-only by wrapping __setitem__
        new._read_only = True

        return new

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
        if self._temp_dir is None:
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

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """ Validate the image and return it. """
        image_size = image.shape[:2]
        if self.reference_size is None:
            self.reference_size = image_size
            self.n_dims = image.ndim
        elif self.reference_size != image_size:
            raise ValueError(f"Image size {image_size} does not match reference size {self.reference_size}.")
        if self.dtype is None:
            self.dtype = image.dtype
            self._update_drange()
        else:
            if self.dtype != image.dtype:
                raise ValueError(f"Image dtype {image.dtype} does not match collection dtype {self.dtype}.")
        return image

    def _to_gray(self, image: np.ndarray):
        gray_map = {1: 'equal', 2: 'rec601', 3: 'rec709', 4: 'rec2020'}
        if image.ndim == 3:
            if self.as_gray > 0:
                image = rgb2gray(image, conversion_type=gray_map[self.as_gray])
                image = uint8_plus(image)
        return image

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
                self.src_paths = sorted(directory.glob(pattern))
            else:
                self.src_paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*"))

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
                        self._save_image(idx, im, save_dir=self._temp_dir)
                    self._reset_data()  # Data will not be stored in self.data when conserve_memory is True
                    self.data[0] = self._validate_image(self._load_image(self.store_paths[0]))
                else:
                    self.has_list_array = True
                    self.data = [self._to_gray(self._validate_image(image)) for image in input_data]
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
                self.data[0] = self._validate_image(self._load_image(self.store_paths[0]))
            elif not self.data and all([isinstance(fp, (str, Path)) for fp in self.store_paths]):
                # Load all images into self.data --- This could not happen:
                self.data = [self._validate_image(self._load_image(fpath)) for fpath in self.store_paths]
            elif not all([isinstance(d, np.ndarray) for d in self.data]):
                raise ValueError('Input data should be either a list of np.ndarray or a glob pattern or a list of Path.')
        self.reference_size = self.data[0].shape[:2]
        self.n_dims = self.data[0].ndim

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
        try:
            if image_path.suffix == ".npy":
                image = np.load(image_path)
                self.dtype = image.dtype
            else:
                with Image.open(image_path) as pil_image:
                    # Load as RGB and convert to grayscale if required
                    pil_image = pil_image.convert(self.DEFAULT_COLOR_MODE)
                    image = np.array(pil_image)
                    image = self._to_gray(image)

                self.dtype = image.dtype
            self._update_drange()
        except IOError as e:
            raise IOError(f"Failed to load image from {image_path}: {e}")

        return image

    def _save_image(self, idx: int, image: np.ndarray, save_dir: Optional[Path] = None) -> None:
        """ Save an image to the temporary directory. """
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
            self.store_paths[idx] = image_path # Update file path
            try:
                if file_format == '.npy':
                    np.save(image_path, image.squeeze())
                else:
                    image = Image.fromarray(image.squeeze())
                    image.save(image_path, format=file_format)
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
        return format_mapping.get(ext, 'TIFF')

    def final_save_all(self) -> None:
        """ Save images to save_dir. If needed (self.conserve_memory) loads images and clears up temp files. """
        for idx in range(self.n_images):
            self._save_image(idx, self[idx], save_dir=self.save_dir)

        # Clean up temporary directory
        self._cleanup_temp_dir()

    def close(self) -> None:
        self._cleanup_temp_dir()

    def __del__(self) -> None:
        """ Clean up temporary directory upon object destruction. """
        self._cleanup_temp_dir()
