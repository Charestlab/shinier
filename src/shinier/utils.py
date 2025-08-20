# from datetime import datetime
import hashlib
from pathlib import Path


# TODO: refactor class and function names; refactor docstrings to google style; get rid of cv2

from numpy.lib.stride_tricks import sliding_window_view
from typing import Any, Optional, Tuple, Union, NewType, List, Iterator, Callable
from PIL import Image, ImageFilter
import shutil
import tempfile
from itertools import chain
import numpy as np
from shinier import Options


# Type definition
ImageListType = Union[str, Path, List[Union[str, Path]], List[np.ndarray]]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
        as_gray (Optional[bool]): If True, images are converted to grayscale upon loading.
            Defaults to False.
        save_dir (Optional[str]): Directory to save final images. Defaults to the
            current working directory if not specified.

    Attributes:
        data: The list of images.
        file_paths: The list of file paths or identifiers.
        reference_size: Reference image size (x, y) for validation.
        n_dims: Number of dimensions of the image
    """

    def __init__(
        self,
        input_data: ImageListType,
        conserve_memory: bool = True,
        as_gray: bool = False,
        save_dir: Optional[str] = None
    ) -> None:
        self.conserve_memory: bool = conserve_memory
        self.as_gray: bool = as_gray
        self.save_dir: Path = Path(save_dir or Path.cwd())
        self.data: List[Optional[np.ndarray]] = []
        self.file_paths: List[Path] = []
        self.reference_size: Optional[Tuple[int, int]] = None
        self.n_dims: Optional[int] = None
        self._temp_dir = Path(tempfile.mkdtemp()) if self.conserve_memory else None
        self.dtype: Optional[type] = None  # Initial state when loading images
        self.drange: Optional[tuple] = None
        self._initialize_collection(input_data)

    def __getitem__(self, idx: int) -> np.ndarray:
        """ Access an image by index. """
        if idx < -len(self.file_paths) or idx >= len(self.file_paths):
            raise IndexError("Index out of range.")
        if idx < 0:
            idx = len(self.file_paths) + idx
        if self.data[idx] is None:
            if self.conserve_memory:
                # If conserve memory, keep only one image in memory
                self._reset_data()
                self.data[idx] = self._validate_image(self._load_image(self.file_paths[idx]))
            else:
                raise ValueError(f"Data at index {idx} is None. This should not happen when conserve_memory is False.")
        return self.data[idx]

    def __setitem__(self, idx: int, new_image: np.ndarray) -> None:
        """ Modify an image at a given index. """
        if idx < 0 or idx >= len(self.file_paths):
            raise IndexError("Index out of range.")

        self.dtype = new_image.dtype
        self._get_drange()
        new_image = self._validate_image(new_image)
        if self.conserve_memory:
            self._reset_data()
            self._save_image(idx, new_image, save_dir=self._temp_dir)
        self.data[idx] = new_image

    def __len__(self) -> int:
        """ Get the number of images in the collection. """
        return len(self.file_paths)

    def __iter__(self) -> Iterator[np.ndarray]:
        """ Iterate over the images in the collection. """
        for idx in range(len(self.file_paths)):
            yield self[idx]

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """ Validate the image and return it. """
        image_size = (image.shape[1], image.shape[0])
        if self.reference_size is None:
            self.reference_size = image_size
            self.n_dims = image.ndim
        elif self.reference_size != image_size:
            raise ValueError(f"Image size {image_size} does not match reference size {self.reference_size}.")
        if self.dtype is None:
            self.dtype = image.dtype
            self._get_drange()
        else:
            if self.dtype != image.dtype:
                raise ValueError(f"Image dtype {image.dtype} does not match collection dtype {self.dtype}.")
        return image

    def _initialize_collection(self, input_data: ImageListType) -> None:
        """ Initialize the image collection from input data. """
        if isinstance(input_data, (str, Path)):
            # Convert to Path if input_data is a string
            input_path = Path(input_data)

            # Handle cases with wildcards
            if "*" in str(input_path):  # Check if it's a glob pattern
                directory = input_path.parent
                pattern = input_path.name
                self.file_paths = sorted(directory.glob(pattern))
            else:
                self.file_paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*"))

            if not self.file_paths:
                raise FileNotFoundError(f"No files found matching pattern '{input_data}'")

        elif isinstance(input_data, list):
            if all(isinstance(item, np.ndarray) for item in input_data):
                if self.conserve_memory:
                    # Save images to temp folder as .npy files
                    self.file_paths = []
                    for idx, image in enumerate(input_data):
                        image = self._validate_image(image)
                        base_name = f'image_{idx}.npy'
                        image_path = self._temp_dir / base_name
                        np.save(image_path, image)
                        self.file_paths.append(image_path)
                    self._reset_data()
                else:
                    self.data = [self._validate_image(image) for image in input_data]
                    if self.file_paths.__len__() == 0:
                        self.file_paths = [None] * len(input_data)
            elif all(isinstance(item, (str, Path)) for item in input_data):
                self.file_paths = [Path(item) for item in input_data]
            else:
                raise TypeError("input_data must be a file pattern, list of file paths, or list of NumPy arrays.")
        else:
            raise TypeError("input_data must be a file pattern, list of file paths, or list of NumPy arrays.")

        if not self.data or all(d is None for d in self.data):
            if self.conserve_memory:
                # Only load the first image to initialize attributes
                self._reset_data() # Data will not be stored in self.data when conserve_memory is True
                self.data[0] = self._validate_image(self._load_image(self.file_paths[0]))
            else:
                # Load all images into self.data
                self.data = [self._validate_image(self._load_image(fpath)) for fpath in self.file_paths]
        self.reference_size = self.data[0].shape[:2]
        self.n_dims = self.data[0].ndim

    def _get_drange(self):
        if self.dtype == np.bool or self.dtype == bool :
            self.drange = (0, 1)
        elif self.dtype == np.uint8:
            self.drange = (0, 2 ** 8 - 1)
        elif self.dtype == np.uint16:
            self.drange = (0, 2 ** 16 - 1)
        elif self.dtype == np.uint32:
            self.drange = (0, 2 ** 32 - 1)
        elif self.dtype == np.uint64:
            self.drange = (0, 2 ** 64 - 1)

    def _load_image(self, image_path: Path) -> np.ndarray:
        """ Load an image from a file path. """
        try:
            if image_path.suffix == ".npy":
                image = np.load(image_path)
                self.dtype = image.dtype
            else:
                self.dtype = np.uint8
                with Image.open(image_path) as image:
                    if self.as_gray:
                        image = image.convert('L')
                    else:
                        image = image.convert('RGB')
            self._get_drange()
        except IOError as e:
            raise IOError(f"Failed to load image from {image_path}: {e}")

        return np.array(image)

    def _save_image(self, idx: int, image: np.ndarray, save_dir: Optional[Path] = None) -> None:
        """ Save an image to the temporary directory. """
        save_dir = Path(save_dir or self._temp_dir or self.file_paths[idx].parent or Path.cwd())
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            base_name = self.file_paths[idx].name
            image_path = save_dir / base_name
            file_format = self._get_file_format(image_path)
            self.file_paths[idx] = image_path # Update file path
            try:
                if file_format == '.npy':
                    np.save(image_path, image.squeeze())
                else:
                    image = Image.fromarray(image.squeeze())
                    image.save(image_path, format=file_format)
            except (IOError, TypeError) as e:
                raise IOError(f"Failed to save image at index {idx} to {image_path}: {e}")
        except (AttributeError) as e:
            raise AttributeError(f"Failed to save image at index {idx}: {e}")

    def _reset_data(self) -> None:
        """ Reset data attribute with placeholders. """
        self.data = [None] * len(self.file_paths)

    @staticmethod
    def _get_file_format(image_path: Path) -> str:
        """ Get the file format based on the file extension. """
        ext = image_path.suffix.lower()
        format_mapping = {
            '.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG',
            '.bmp': 'BMP', '.tiff': 'TIFF', '.tif': 'TIFF', '.npy': '.npy'
        }
        return format_mapping.get(ext, 'TIFF')

    def final_save_all(self) -> None:
        """ Save images to save_dir. If needed (self.conserve_memory) loads images and clears up temp files. """
        for idx in range(len(self.file_paths)):
            image = self._load_image(self._temp_dir / f'image_{idx}.npy') if self.conserve_memory else self[idx]
            self._save_image(idx, image, save_dir=self.save_dir)

        # Clean up temporary directory
        self._cleanup_temp_dir()

    def _cleanup_temp_dir(self) -> None:
        """ Clean up temporary directory if it exists. """
        # if self._temp_dir and self._temp_dir.is_dir():
        try:
            if self._temp_dir:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self._temp_dir = None
        except Exception as e:
            # Log or handle the exception appropriately
            print(f"Failed to delete temporary directory {self._temp_dir}: {e}")
    
    def close(self):
        self.__del__()
    
    def __del__(self):
        """ Clean up temporary directory upon object destruction. """
        self._cleanup_temp_dir()


class MatlabOperators:

    @staticmethod
    def round(x):
        """
         alt_round(x)

         <x> is an array

         simulate the rounding behavior of matlab where 0.5 rounds
         to 1 and -.5 rounds to -1. (python rounds ties to the
         nearest even integer.)

         return:
          an array of rounded values

         example:
         import numpy as np
         x = np.array([-1, -0.5, 0, 0.5, 0.7, 1.0, 1.5, 2.1, 2.5, 2.6, 3.5])
         y = alt_round(x)

         from https://github.com/cvnlab/GLMsingle/blob/main/glmsingle/utils/alt_round.py
        
        return (np.sign(x) * np.ceil(np.floor(np.abs(x) * 2) / 2)).astype(int)
        ----
        Slight modifications to follow MATLAB's behavior of not changing types: Mathias Salvas-Hébert, 2025-07-16
         MATLAB : 
            x = int8([-3.7, -1.2, 0.5, 2.9, 10.1]);
            y = round(x);   
            class(y) 
            > ans = 'int8'

            x = double([-3.7, -1.2, 0.5, 2.9, 10.1]);
            y = round(x);
            class(y) 
            > ans = 'double'
        """
        return np.sign(x) * np.ceil(np.floor(np.abs(x) * 2) / 2)

    @staticmethod
    def uint8(x):
        """Replicates MATLAB's uint8 behavior: rounds and clips to [0, 255]"""
        return np.uint8(np.clip(MatlabOperators.round(x), 0, 255))

    @staticmethod
    def std2(A):
        """Replicates MATLAB's std2 function, which uses ddof=1"""
        return np.std(A, ddof=1)

    @staticmethod
    def mean2(A):
        return np.mean(A)
    @staticmethod
    def double(x):
        """Ensures double precision, similar to MATLAB's double."""
        return np.array(x, dtype=np.float64)

    @staticmethod
    def single(x):
        """Ensures single precision, similar to MATLAB's single."""
        return np.array(x, dtype=np.float32)

    @staticmethod
    def int16(x):
        """Replicates MATLAB's int16 behavior: rounds and clips to [-32768, 32767]"""
        return np.int16(np.clip(MatlabOperators.round(x), -32768, 32767))

    @staticmethod
    def int32(x):
        """Replicates MATLAB's int32 behavior: rounds and clips to [-2**31, 2**31 - 1]"""
        return np.int32(np.clip(MatlabOperators.round(x), -2**31, 2**31 - 1))

    @staticmethod
    def linspace(start, stop, num=50):
        """Replicates MATLAB's linspace function."""
        return np.linspace(start, stop, num)

    @staticmethod
    def ceil(x):
        """Replicates MATLAB's ceil function."""
        return np.ceil(x)

    @staticmethod
    def floor(x):
        """Replicates MATLAB's floor function."""
        return np.floor(x)

    @staticmethod
    def fix(x):
        """Replicates MATLAB's fix function (rounds toward zero)."""
        return np.sign(x) * np.floor(np.abs(x))

    @staticmethod
    def mod(x, y):
        """Replicates MATLAB's mod function (modulus operation with sign matching divisor)."""
        return np.mod(x, y)

    @staticmethod
    def rem(x, y):
        """Replicates MATLAB's rem function (remainder operation with sign matching dividend)."""
        return np.remainder(x, y)


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Efficiently convolve a given 2D image with a given square kernel.

    Args:
        image (np.ndarray): Input 2D image.
        kernel (np.ndarray): Square convolution kernel.

    Returns:
        np.ndarray: Convolved image.
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError('Kernel must be a 2D np.ndarray of square shape')
    if not isinstance(image, np.ndarray):
        raise TypeError('Image must be a 2D np.ndarray')
    if image.ndim == 3:
        raise TypeError('Image must be a 2D np.ndarray')
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError('Kernel must be a 2D np.ndarray of square shape')

    # Get kernel dimensions (assuming square kernel)
    kernel_size = kernel.shape[0]

    # Apply reflective padding to the image
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')

    # Create sliding window view
    windows = sliding_window_view(padded_image, (kernel_size, kernel_size))

    # Perform convolution using optimized summation
    conv_result = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
    return conv_result


def pixel_order(image: np.ndarray) -> Tuple[np.ndarray, Union[float, list]]:
    """ 
    Assigns strict ordering to monochromatic or multispectralimage pixels.

    Args:
        image (np.ndarray): image

    Returns:
        Tuple[np.ndarray, Union[float, list]]:
            - im_sort (np.ndarray): Image with same dimensions as input, with elements representing the pixel order.
            - OA (float or list): Order accuracy in the range [0, 1], fraction of unique filter response combinations.
              For multichannel images, OA is a list.
    """
    M, N, P = image.shape

    # Defining the 6 filters (F1 = grayscale of pixel for a channel)
    F2 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) / 5.0
    F3 = np.ones((3, 3)) / 9.0
    F4 = np.ones((5, 5)) / 13.0
    F4[[0, 0, 1, 1, 1, 3, 3, 4, 4, 4], [0, 1, 0, 1, 4, 0, 4, 1, 3, 4]] = 0
    F5 = np.ones((5, 5)) / 21.0
    F5[[0, 0, 4, 4], [0, 4, 0, 4]] = 0
    F6 = np.ones((5, 5)) / 25.0
    
    # Filters ordered by importance  
    F = [F2, F3, F4, F5, F6]

    # Convolve filters with the image and order
    im_sort = []
    OA = []

    for i in range(P):
        # Apply filters to each channel and collect filter responses
        FR = np.zeros((M, N, 6))
        FR[:, :, 0] = image[:, :, i]
        for j in range(5):
            FR[:, :, j + 1] = convolve_2d(image[:, :, i], F[j])

        # Rearrange the filter responses
        FR = FR.reshape(M * N, 6)

        # Number of unique filter responses and ordering accuracy
        unique_responses = np.unique(FR, axis=0)
        n = unique_responses.shape[0]
        OA.append(n / (M * N))

        # Sort responses lexicographically
        # [:, ::-1] because np.lexsort applies sort keys from last to first (right to left).
        idx_pos = np.lexsort(FR[:, ::-1].T)

        # Rearrange indices according to pixel position
        idx_o = np.argsort(idx_pos)
        idx_o = idx_o.reshape((M, N))

        im_sort.append(idx_o)

    if P == 1:
        OA = OA[0]

    return np.stack(im_sort, axis=-1), OA


def exact_histogram(image: np.ndarray, target_hist: np.ndarray, binary_mask: np.ndarray = None, verbose: bool = True) -> Tuple[np.ndarray, List]:
    """
    Specify exact image histogram.

    Args:
        image (np.ndarray): Input image (8-bit or 16-bit grayscale or RGB).
        target_hist (np.ndarray): Specified histogram.
        binary_mask (np.ndarray): Binary mask to only adjust pixel intensities in the foreground (optional).
        verbose (bool): Log information

    Returns:
        Tuple[np.ndarray, Union[float, list]]:
            - im_out (np.ndarray): Processed image with specified histogram.
            - OA (list): Order accuracy indicating fraction of unique filter response combinations.

    References:
        1. Coltuc D. and Bolon P., 1999, "Strict ordering on discrete images
        and applications"
        2. Coltuc D., Bolon P. and Chassery J-M., 2006, "Exact histogram
        specification", IEEE Transactions on Image Processing
        15(5):1143-1152

    This code is a Python implementation of Anton Semechko's (asemechk@uoguelph.ca) exact_histogram in MATLAB
    """

    # Maximum number of gray levels and image dimensions
    L = 2 ** np.iinfo(image.dtype).bits if np.issubdtype(image.dtype, np.integer) else None
    image = im3D(image) if image.ndim != 3 else image  # force a third dimension on image
    x_size, y_size, n_channels = image.shape

    # Verify input format
    if not image.dtype in (np.uint8, np.uint16):
        raise ValueError("Input image must be 8- or 16-bit.")
    if len(target_hist) != L:
        raise ValueError("Number of histogram bins must match maximum number of gray levels.")
    if target_hist.ndim != 2 or target_hist.shape[1] != n_channels:
        raise ValueError("Target histogram (target_hist) should have the same number of channels as the image.")
    if binary_mask is not None:
        binary_mask = im3D(binary_mask) if binary_mask.ndim == 2 else binary_mask  # force a third dimension on image
        if not image.shape == binary_mask.shape:
            raise ValueError(f"binary_mask shape ({binary_mask.shape}) should be equal to image shape ({image.shape})")
        if np.sum(binary_mask) < (50 * n_channels):
            raise ValueError("Too few foreground pixels in the binary mask.")
    else:
        binary_mask = np.ones(image.shape, dtype=bool)
    if n_channels not in [1, 3]:
        raise ValueError("Input image must have 1 or 3 channels.")

    # Assign strict order to pixels
    im_sort, OA = pixel_order(image)

    # Process each channel separately
    im_out = image.copy()
    for channel in range(n_channels):
        # Work only on the masked (foreground) pixels
        foreground_indices = binary_mask[:, :, channel]
        Ntotal = np.sum(foreground_indices)

        # Get the pixel order values for the masked region
        pix_ord = im_sort[:, :, channel][foreground_indices]

        # Sort pixel order and get the sorted indices
        sorted_indices = np.argsort(pix_ord)

        # Adjust the specified histogram to match the number of pixels in the mask
        new_target_hist = (Ntotal * target_hist[:, channel] / np.sum(target_hist[:, channel])).astype(int)
        residuals = Ntotal - np.sum(new_target_hist)
        # Redistribute the residuals to ensure total counts match
        sorted_residuals = np.argsort(-np.mod(Ntotal * target_hist[:, channel] / np.sum(target_hist[:, channel]), 1))
        new_target_hist[sorted_residuals[:residuals]] += 1

        # Create intensity values based on the adjusted histogram
        Hraw = np.repeat(np.arange(L), new_target_hist)

        # Reorder Hraw according to the sorted pixel positions
        Hraw_sorted = np.zeros_like(Hraw)
        Hraw_sorted[sorted_indices] = Hraw

        # Assign the sorted intensity values back to the output image
        im_out[:, :, channel][foreground_indices] = Hraw_sorted.astype(image.dtype)
        im_out[:, :, channel][~foreground_indices] = image[:, :, channel][~foreground_indices]

    return im_out.squeeze(), OA


def noisy_bit_dithering(image: np.ndarray, depth: int = 256) -> np.ndarray:
    """
    Implements the dithering algorithm presented in :
        Allard, R., Faubert, J. (2008) The noisy-bit method for digital displays:
        converting a 256 luminance resolution into a continuous resolution. Behavior
        Research Method, 40(3), 735-743.

    Args:
        image (np.ndarray): An image of floats ranging from 0 to 1.
        depth (optional) : The number of gray shades. (Default = 256)

    Returns:
        processed_image (np.ndarray): image matrix containing integer values [1, depth], indicating which luminance value should be used for every pixel.
            Output uses the smallest integer dtype that fits all values.
    E.g.:
        processed_image = noisy_bit_dithering(image, depth = 256)

    This example assumes that all rgb values are linearly related to luminance
    values (e.g. on a Mac, put your LCD monitor gamma parameter to 1 in the Displays
    section of the System Preferences). If this is not the case, use a lookup table
    to transform the tim integer values into rgb values corresponding to evenly
    spaced luminance values.

    Frederic Gosselin, 27/09/2022
    frederic.gosselin@umontreal.ca

    Slight modifications for Matlab compatibility: Nicolas Dupuis-Roy & Mathias Salvas-Hébert, 2025-08-19

    """
    if not isinstance(image, np.ndarray) or np.issubdtype(image.dtype, np.integer):
        raise TypeError('image should be a np.ndarray of floats ranging from 0 to 1')
    if not isinstance(depth, int):
        raise TypeError('depth should be an integer')

    processed_image = image * (depth - 1.0)

    # tim = np.uint8(np.fmax(np.fmin(np.around(tim + np.random.random(np.shape(im)) - 0.5), depth - 1.0), 0.0))
    processed_image = processed_image + np.random.random(np.shape(image)) - 0.5
    tim = np.clip(MatlabOperators.round(processed_image), 0, depth-1)
    uint_image = tim.astype(np.min_scalar_type(int(tim.max()))).squeeze()
    return uint_image


def uint_to_float01(image: np.ndarray, allow_clipping: bool = True) -> np.ndarray:
    """
    Convert an N-bit unsigned integer (uintN) image to a floating-point image with values ranging from 0 to 1.

    A float is assumed to be within the [0, 1] range whereas the uintN within the [0, n_levels] range.

    Args:
        image (np.ndarray): Input image as a NumPy array with floating-point values.
        allow_clipping (bool): If True, clip values outside the range [0, 1].
                               If False, raises an error if values are out of range.

    Returns:
        np.ndarray: The converted image as a NumPy array with dtype float64.

    Raises:
        ValueError: If `allow_clipping` is False and the image contains values outside the range [0, 1].
    """
    if not isinstance(image, np.ndarray) or not np.issubdtype(image.dtype, np.integer):
        raise TypeError('image should be a np.ndarray of integers')

    # Determine the range of the input image
    image_min, image_max = image.min(), image.max()

    # Scale image if range is [0, 1]
    n_levels = np.iinfo(image.dtype).max
    image = image.astype(np.float64) / n_levels

    # Check if values are within [0, 255]
    if not allow_clipping and (image_min < 0 or image_max > 1):
        raise ValueError("Image contains values outside the range [0, 1]. Consider enabling clipping.")

    # Clip values if allowed
    image_clipped = np.clip(image, 0, 1) if allow_clipping else image

    # Convert to uint8
    return image_clipped.astype(np.float64)


def float01_to_uint(image: np.ndarray, allow_clipping: bool = True, bit_size: int = 8) -> np.ndarray:
    """
    Convert a floating-point image to an n-bit unsigned integer (uintN) image.

    A float is assumed to be within the [0, 1] range whereas the uintN within the [0, n_levels] range.

    Args:
        image (np.ndarray): Input image as a NumPy array with floating-point values.
        allow_clipping (bool): If True, clip values outside the range [0, n_levels].
                               If False, raises an error if values are out of range.
        bit_size (int): Bit size of the unsigned integer.

    Returns:
        np.ndarray: The converted image as a NumPy array with dtype uintN.

    Raises:
        ValueError: If `allow_clipping` is False and the image contains values outside the range [0, n_levels].
    """

    if not isinstance(image, np.ndarray) or not np.issubdtype(image.dtype, np.floating):
        return image

    # Define a mapping from bit_size to NumPy unsigned integer types
    uint_types = {
        8: np.uint8,
        16: np.uint16,
        32: np.uint32,
        64: np.uint64,
    }

    if bit_size not in uint_types:
        raise ValueError(f"Invalid bit size {bit_size}. Supported values are {list(uint_types.keys())}.")

    # Get the target dtype
    target_dtype = uint_types[bit_size]

    # Clip the input image to fit within the range of the target dtype
    dtype_info = np.iinfo(target_dtype)

    n_levels = dtype_info.max
    image = image * n_levels

    # Check if values are within [0, n_levels]
    if not allow_clipping and (image.min() < 0 or image.max() > n_levels):
        raise ValueError(f"Image contains values outside the range [0, {n_levels}]. Consider enabling clipping.")

    # Clip values if allowed
    image_clipped = np.clip(image, dtype_info.min, dtype_info.max) if allow_clipping else image

    # Convert to uint8
    return image_clipped.astype(target_dtype)


def pol2cart(magnitude: np.ndarray, angle: np.ndarray) -> tuple:
    """
    Convert polar coordinates (magnitude, angle) to Cartesian coordinates (x, y).

    Args:
        magnitude (np.ndarray): Distance from the origin (radius).
        angle (np.ndarray): Angle in radians.

    Returns:
        tuple: A tuple (x, y) where:
            - x is the Cartesian x-coordinate.
            - y is the Cartesian y-coordinate.
    """
    x = magnitude * np.cos(angle)  # Compute x-coordinate
    y = magnitude * np.sin(angle)  # Compute y-coordinate
    return x, y


def cart2pol(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (magnitude, angle).

    Args:
        x (np.ndarray): Real part (x-coordinate).
        y (np.ndarray): Imaginary part (y-coordinate).

    Returns:
        tuple: A tuple (magnitude, angle) where:
            - magnitude is the distance from the origin.
            - angle is the angle in radians.
    """
    magnitude = np.sqrt(x**2 + y**2)  # Compute magnitude
    angle = np.arctan2(y, x)          # Compute angle
    return magnitude, angle


def rgb2gray(image: Union[np.ndarray, Image.Image], conversion_type: Union[str] = 'perceptual') -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    Args:
        image (Union[np.ndarray, Image.Image]): The input image.
        conversion_type (str): If 'perceptual' a weighted sum of the corresponding red, green and blue
            pixels is applied to better represent human perception of red, green and blue than equal weights. Else,
            Equal weighted sum is applied. See http://poynton.ca/PDFs/ColorFAQ.pdf
    Returns:
        np.ndarray: The grayscale image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if image.ndim > 2:
        if conversion_type == 'perceptual':
            weights = [0.2125, 0.7154, 0.0721]
        else:
            weights = [1/3, 1/3, 1/3]
        return np.dot(image[..., :3].astype(np.float32), weights)
    elif image.ndim == 2:
        return image

def gray2rgb(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convert a grayscale image to RGB.

    Args:
        image (Union[np.ndarray, Image.Image]): The input grayscale image.

    Returns:
        np.ndarray: The RGB image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32)

    if image.ndim > 2:
        image = image[:, :, 0]

    return np.stack((image,) * 3, axis=-1)


def separate(mask: np.ndarray, background: Union[int, float] = None, smoothing: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Function for simple figure-ground segregation.
    Args:
      mask (np.ndarray): Source mask. Could be an image or a bit mask.
        background (Optional[Union[uint8, float64]]); uint8 value of the background ([0,255]) (e.g., 255) 
        or float64 value of the background ([0,1]) (e.g., 1); if equals to 300 or not specified, it is the
        value that occurs the most frequently in mask.
      smoothing (bool): If true, applies median blur on mask.

    Returns:
        mask_fgr (np.ndarray[bool]): 2D matrix of the same size as the source mask; Foreground is True
            and background is False.
        mask_bgr (np.ndarray[bool]): 2D matrix of the same size as the source mask; Background is True
            and foreground is False.
        background (Optional[np.uint8]): Specifies the value that was used to
            define the background in the original image

    """

    mask = rgb2gray(mask)
    mask = mask.astype(np.float64)/255 if np.max(mask) > 1 else mask
    background = background/255 if background > 1 and background < 256 else background 

    if background == 300:
        # Use np.unique to get unique values and their counts
        unique_values, counts = np.unique(mask.flatten(), return_counts=True)
        background = unique_values[np.argmax(counts)]

    mask_bgr = mask == background

    # Apply median filter to smooth the mask
    mask_bgr = apply_median_blur(mask_bgr)
    mask_fgr = (mask_bgr * -1) + 1

    # TODO: add display options?
    # if fig:
    #     plt.figure(figsize=(10, 5))
    #
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(mask_fgr, cmap='gray')
    #     plt.title('Foreground Mask')
    #
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(mask_bgr, cmap='gray')
    #     plt.title('Background Mask')
    #
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(orig_im, cmap='gray')
    #     plt.title('Original Image')
    #     plt.show()

    return mask_fgr.astype(bool), mask_bgr.astype(bool), background



def image_spectrum(image: np.ndarray, rescale: bool = True) -> np.ndarray:
    """
    Compute spectrum of an image
    Args:
        image (np.ndarray): An image
        rescale (bool): If true, will rescale each channel to [0, 1] range.

    Returns:
        magnitude, phase

    """
    image = im3D(image)
    if rescale:
        image = rescale_image(image)  # [0, 255] -> [0, 1]

    x_size, y_size, n_channels = image.shape
    phase = np.zeros((x_size, y_size, n_channels))  # Phase FT
    magnitude = np.zeros((x_size, y_size, n_channels))  # Magnitude FT

    image = im3D(image)
    for channel in range(image.shape[-1]):
        fft_image = np.fft.fftshift(np.fft.fft2(image[:, :, channel]))
        magnitude[:, :, channel], phase[:, :, channel] = cart2pol(np.real(fft_image), np.imag(fft_image))
    return magnitude, phase


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.

    Args:
        size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Normalized 2D Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def ssim_sens(image1: np.ndarray, image2: np.ndarray, n_bins: int = 256) -> Tuple[np.ndarray, float]:
    """
    Compute the Structural Similarity Index (SSIM) and its gradient.

    Args:
        image1 (np.ndarray): First image as a 3D array.
        image2 (np.ndarray): Second image as a 3D array.
        n_bins (int, optional): Dynamic range of pixel values. Defaults to 256.

    Returns:
        Tuple[np.ndarray, float]:
            - Gradient of SSIM (sensitivity) as a 2D array.
            - Mean SSIM value as a float.

    References : 
        1. Avanaki, A.N. Exact global histogram specification optimized for structural similarity. 
        OPT REV 16, 613–621 (2009). https://doi.org/10.1007/s10043-009-0119-z

        2. Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: 
        from error visibility to structural similarity," in IEEE Transactions on Image Processing,
        vol. 13, no. 4, pp. 600-612, April 2004, doi: 10.1109/TIP.2003.819861.
    """
    image_x_3D = im3D(image1.astype(np.float64))
    image_y_3D = im3D(image2.astype(np.float64))

    # Gaussian kernel parameters
    window_size = 11
    sigma = 1.5
    window = gaussian_kernel(window_size, sigma)

    # Constants for SSIM
    L = n_bins - 1
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Mean calculations
    all_sens, all_mssim = [], []
    for channel in range(image_x_3D.shape[2]):
        # Select channels
        image_x = image_x_3D[:, :, channel]
        image_y = image_y_3D[:, :, channel]

        # Mean pixel intensity
        mu_x = convolve_2d(image_x, window)
        mu_y = convolve_2d(image_y, window)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_x_y = mu_x * mu_y

        # Variances et covariance
        sigma_x_sq = convolve_2d(image_x ** 2, window) - mu_x_sq
        sigma_y_sq = convolve_2d(image_y ** 2, window) - mu_y_sq
        sigma_x_y = convolve_2d(image_x * image_y, window) - mu_x_y

        # SSIM map (Eq. 6)
        num_1 = 2 * mu_x_y + C1
        num_2 = 2 * sigma_x_y + C2
        num = (num_1 * num_2) 

        den_1 = mu_x_sq + mu_y_sq + C1
        den_2 = sigma_x_sq + sigma_y_sq + C2
        den = (den_1 * den_2) 

        ssim_map = num / den
        mssim = np.mean(ssim_map)
        
        # SSIM gradient - Eqs. (7) and (8) (Avanaki, 2009)
        term_1 = num_1 / den
        sens = convolve_2d(term_1, window) * image_x

        term_2 = -ssim_map/den_2
        sens += convolve_2d(term_2, window) * image_y

        term_3 = (mu_x * (num_2 - num_1) - mu_y * ssim_map * (den_2 - den_1)) / den
        sens += convolve_2d(term_3, window)

        sens *= 2 / sens.size 
        
        # FIX : scales the SSIM gradient to compensate for its attenuation at larger n_bins and keep the
        # effective update weight consistent across bit depths (more bins require proportionally larger 
        # changes for the same effect).
        sens *= 256**(2*(np.log(n_bins) / np.log(256)-1))

        all_sens.append(sens)
        all_mssim.append(mssim)
    return np.stack(all_sens, axis=-1).squeeze(), np.stack(all_mssim)

def compute_rmse(image1: np.ndarray, image2: np.ndarray) -> float:
    """ Compute the root-mean-square error between two images. """
    return np.sqrt(np.mean((image1 - image2) ** 2))

def compute_metrics_from_paths(images: ImageListType, options: Options):
    """Computes the average SSIM and RMSM between the original images and the processed ones

    Args:
        images (ImageListType): images after being processed
        options (Options):
            as_gray (bool): If the input images and RGB or grayscale. If True, images are converted to grayscale upon loading.
            input_folder (Union[str, Path]): relative or absolute path of the image folder (default = ./INPUT)
            images_format (str): png, tif, jpg (default = 'tif')
            metrics (List[str], optional): Metrics to compute. Defaults to ['rmse', 'ssim'].

    Returns:
        output (dict): with 'avg_rmse' and 'avg_ssim' if computed.
    """
    if options.metrics != None : 
        total_rmse = 0
        total_ssim = 0
        output = []
        image_paths = Path(options.input_folder) / f"*.{options.images_format}"
        directory = image_paths.parent
        pattern = image_paths.name
        file_paths = sorted(directory.glob(pattern))

        for image_path, proc_im in zip(file_paths, images.images):
            # 1 : Load original image
            with Image.open(image_path) as image:
                if options.as_gray:
                    orig_im = np.array(image.convert('L'))
                else:
                    orig_im = np.array(image.convert('RGB'))

            # 2 : Compute the metrics
            if 'rmse' in options.metrics:
                rmse_value = compute_rmse(orig_im, proc_im)
                total_rmse += rmse_value
            if 'ssim' in options.metrics:
                _, ssim_value = ssim_sens(orig_im, proc_im)
                total_ssim += ssim_value.squeeze()
        
        if 'rmse' in options.metrics:
            avg_rmse = total_rmse / len(file_paths)
            print(total_rmse, len(file_paths))
            print(f"Average RMSE: {avg_rmse:}")
            output.append(avg_rmse)
        if 'ssim' in options.metrics:
            avg_ssim = total_ssim / len(file_paths)
            print(f"Average SSIM: {avg_ssim}")
            output.append(avg_ssim)
        return output


def get_images_spectra(images: ImageListType, magnitudes: Optional[ImageListIO] = None, phases: Optional[ImageListIO] = None) -> Union[List[np.ndarray], ImageListType]:
    """
    Get spectrum over list of images
    Args:
        images (ImageListType): List of images.
        magnitudes (Optional[ImageListType]): If provided, inserts new magnitudes into this list.
        phases (Optional[ImageListType]): If provided, inserts new phases into this list.
    Returns:
        magnitudes, phases (Union[List[np.ndarray], ImageListType])

    """
    n_images = len(images)
    x_size, y_size = images.reference_size[:2]
    n_channels = 3 if images.n_dims == 3 else 1
    phases = [None] * n_images if phases is None else phases
    magnitudes = [None] * n_images if magnitudes is None else magnitudes
    for idx, image in enumerate(images):
        magnitudes[idx], phases[idx] = image_spectrum(image)
    return magnitudes, phases


def rescale_image(image: np.ndarray, target_min: Optional[float] = 0, target_max: Optional[float] = 1) -> np.ndarray:
    """
    Rescale an image to the range: [target_min, target_max]

    Args:
        image (np.ndarray): Input image
        target_min (float) : Target minimum value in the output image.
        target_max (float) : Target maximum value in the output image.

    Returns:
        (np.ndarray): Rescaled image
    """

    # Convert to float
    image = image.astype(float)

    # Rescale to the specified range: default is [0, 1]
    image -= np.min(image)
    image /= np.max(image)
    image *= (target_max - target_min)
    image += target_min
    return image


def rescale_images(images: ImageListType, rescaling_option: int = 1, legacy_mode: bool = False) -> ImageListType:
    """
    Rescales the values of images so that they fall between 0 and 255. There are 3 options:
        1) Each image has its own min and max (no rescaling)
        2) Each image is rescaled so that the absolute max and min values obtained across all images are between 0 and 255
        3) Each image is rescaled so that the average max and min values obtained across all images are between 0 and 255

        Args:
            images : list of images
            rescaling (optional) : Determines the type of rescaling.
                0 : No rescaling
                1 : Rescaling absolute max/min (Default)
                2 : Rescaling average max/min

        Returns :
            A list of rescaled images
    """

    # TODO: Shouldn't the input be a uint8 or at least in the [0 to 255] range already?
    if rescaling_option not in [0, 1, 2]:
        raise ValueError(f'The rescaling option must be either [0, 1, 2], now rescaling is : {rescaling_option}')

    n_images = len(images)
    minimum_values = np.zeros((n_images,))
    maximum_values = np.zeros((n_images,))
    for idx, image in enumerate(images):
        minimum_values[idx], maximum_values[idx] = np.min(image), np.max(image)

    if rescaling_option == 1:
        mn, mx = np.min(minimum_values), np.max(maximum_values)
    elif rescaling_option == 2:
        mn, mx = np.mean(minimum_values), np.mean(maximum_values)

    if rescaling_option:
        for idx, image in enumerate(images):
            new_image = image.copy().astype(float)
            new_image = (new_image - mn)/(mx - mn) * 255
            images[idx] = MatlabOperators.uint8(new_image) if legacy_mode else np.clip(new_image, 0, 255).astype(np.uint8)

    return images


def apply_median_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a median blur to image.

    Parameters:
        image (np.ndarray): Input image (2D grayscale or 3D multi-channel).
        kernel_size (int): Size of the square kernel (must be an odd integer).

    Returns:
        np.ndarray: Blurred image

    Tested by Nicolas D.R.: Should provide exact same results on np.uint8 as cv2.medianBlur.
    """
    if kernel_size % 2 == 0:
        raise ValueError("median_blur_force must be an odd integer.")
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be a 2D or 3D array.")

    original_ndim = image.ndim
    image_3d = im3D(image)
    pad = kernel_size // 2

    # Pad spatial dimensions only
    padded = np.pad(image_3d, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='edge')

    # Sliding window over H and W only, channels preserved
    windows = sliding_window_view(padded, (kernel_size, kernel_size, 1))
    # Result shape: (H, W, C, k, k, 1)

    # Move (k, k) to the end, then flatten and compute median
    windows = windows.squeeze(-1)  # shape (H, W, C, k, k)
    k = kernel_size
    windows = windows.reshape(*windows.shape[:3], k * k)  # (H, W, C, k*k)

    # Apply filter and restore original dimensions
    blurred = np.median(windows, axis=-1).astype(image.dtype)
    blurred = blurred[:, :, 0].squeeze() if original_ndim == 2 else blurred

    return blurred


def hist2list(hist: np.ndarray) -> np.ndarray:
    """
    Converts a luminance histogram of an image into a sorted list of luminance values.

    Args:
        hist (np.ndarray): Luminance histogram (counts of each intensity) for each channel.

    Returns:
        np.ndarray: Sorted list of luminance values for each channel. It has the same number
            of channels as hist.There is one list per channel and its size is equal to  np.sum(hist[:, channel]).
    """
    if hist.shape[1]>1 and not np.unique(np.sum(hist, axis=1)).shape[0] != 1:
        raise ValueError(f"All 3 channels of hist must have the same sum")
    final_lists = np.zeros((int(np.sum(hist[:, 0])), hist.shape[1]))
    for channel in range(hist.shape[1]):
        final_lists[:, channel] = np.stack(list(chain.from_iterable([val] * cnt for val, cnt in enumerate(hist[:, channel])))).astype(int)
    return final_lists


def im3D(image: np.ndarray):
    """ Forces a third dimension on grayscale image.

    Args:
        image (np.ndarray):

    Returns:
        image (np.ndarray) with 3D

    """
    return np.stack((image,) * 1, axis=-1) if image.ndim != 3 else image


def imhist(image: np.ndarray, mask: Optional[np.ndarray] = None, n_bins=256) -> np.ndarray:
    """ Computes the histogram of the image. If RGB image, it provides one hist per channel.

        Args:
            image (np.ndarray): Image (ndarray).
            mask (np.ndarray): If a boolean mask is provided, computes the histogram within the mask (ndarray).
            n_bins: Number of bins for the histogram (default is 256).
        Returns:
            Y (np.ndarray): Histogram counts for each channel.
            X (np.ndarray): Bin locations for each channel.
    """
    # Force a third dimension to image in case it only has two
    image = im3D(image)

    # If no mask provided, make a blank mask with all True
    mask = np.ones(image.shape).astype(bool) if mask is None else mask.astype(bool)
    mask = np.stack((mask, ) * image.shape[-1], axis=-1) if mask.ndim < image.ndim else im3D(mask)
    n_channels = image.shape[-1]
    count = np.zeros((n_bins, n_channels)).astype(np.int64)
    for channel in range(n_channels):
        count[:, channel], _ = np.histogram(image[:, :, channel][mask[:, :, channel]], bins=n_bins, range=(0, n_bins))
    return count



# Extra utilities :
def floyd_steinberg_dithering(image : np.ndarray, depth : int = 256) -> np.ndarray:
        """
        Implements the dithering algorithm presented in :
            R.W. Floyd, L. Steinberg, An adaptive algorithm for spatial grey scale.
            Proceedings of the Society of Information Display 17, 75Ð77 (1976).
        
        Args:
            image (np.ndarray): An image of floats ranging from 0 to 1.
            depth (optional) : The number of gray shades. (Default = 256)
            
        Returns:
            processed_image (np.ndarray): image matrix containing integer values [1, depth], indicating which luminance value should be used for every pixel.     
                Output uses the smallest integer dtype that fits all values.
        """
        if not isinstance(image, np.ndarray) or np.issubdtype(image.dtype, np.integer):
            raise TypeError('image should be a np.ndarray of floats ranging from 0 to 1')
        if not isinstance(depth, int):
            raise TypeError('depth should be an integer')

        tim = image * (depth - 1.0)
        for xx in np.arange(1,image.shape[1]-1,1):
            for yy in np.arange(1,image.shape[0]-1,1): # exchange with the following
                oldpixel = tim[yy,xx]
                newpixel = MatlabOperators.round(tim[yy,xx])
                quant_error = oldpixel - newpixel
                tim[yy,xx+1] = tim[yy,xx+1] + 7/16 * quant_error
                tim[yy+1,xx-1] = tim[yy+1,xx-1] + 3/16 * quant_error
                tim[yy+1,xx] = tim[yy+1,xx] + 5/16 * quant_error
                tim[yy+1,xx+1] = tim[yy+1,xx+1] + 1/16 * quant_error

        tim = np.clip(MatlabOperators.round(tim), 0, depth-1)
        uint_image = tim.astype(np.min_scalar_type(int(tim.max()))).squeeze()
        return uint_image

def gaussian_ellipsoidal_mask(ims, grayTone=127, RGB=False, cutoffA=0.5, cutoffB=0.75, offsetA=0, offsetB=0):
    """
    Applies a smooth, ellipsoidal Gaussian mask to a list of images.

    The mask is defined by an ellipse with configurable axes and offsets, and is blurred using a Gaussian kernel.
    The masked region blends the image towards a specified gray tone, with optional support for RGB images.

    Args:
        ims : list or array-like
            List of input images (as numpy arrays or PIL Images) to be masked.
        grayTone : int, optional
            The gray tone (0-255) to blend towards in the masked region. Default is 127.
        RGB : bool, optional
            If True, treats images as RGB (3-channel). If False, treats as grayscale. Default is False.
        cutoffA : float, optional
            Semi-axis length of the ellipse along the x-direction (normalized to [0,1]). Default is 0.5.
        cutoffB : float, optional
            Semi-axis length of the ellipse along the y-direction (normalized to [0,1]). Default is 0.75.
        offsetA : float, optional
            Offset of the ellipse center along the x-direction (normalized to [-1,1]). Default is 0.
        offsetB : float, optional
            Offset of the ellipse center along the y-direction (normalized to [-1,1]). Default is 0.

    Returns:
        out : list of PIL.Image
            List of masked images as PIL Image objects, with the ellipsoidal mask applied.

    Notes:
        - The mask smoothly blends the image towards the specified gray tone inside the ellipse.
        - The mask is blurred using a Gaussian kernel for smooth transitions.
        - For RGB images, the mask is applied to all channels.
    """
    def gaussian_blur(a, s=5):
        r = int(3*s)
        x = np.arange(-r, r+1)
        k = np.exp(-0.5*(x/s)**2); k /= k.sum()
        a = np.pad(a, ((0,0),(r,r)), mode='reflect')
        a = np.apply_along_axis(lambda v: np.convolve(v, k, mode='valid'), 1, a)
        a = np.pad(a, ((r,r),(0,0)), mode='reflect')
        a = np.apply_along_axis(lambda v: np.convolve(v, k, mode='valid'), 0, a)
        return a
    out = []
    for im in ims:
        im = np.asarray(im); H, W = im.shape[:2]; f = im.astype(np.float64)/255.0
        xv, yv = np.meshgrid(np.linspace(0,1,W), np.linspace(0,1,H))
        m = ((((2*xv-1-offsetA)**2)/(cutoffA**2) + ((2*yv-1-offsetB)**2)/(cutoffB**2)) < 1).astype(np.float64)
        m = gaussian_blur(m); mx = m.max(); m = m/(mx if mx>0 else 1.0)
        f = m[...,None]*(f-0.5)+0.5 if (RGB and f.ndim==3 and f.shape[2]==3) else m*(f-0.5)+0.5
        a = f*255.0; g = 127 - grayTone
        o = np.where(a!=255.0, a-g, 255.0)
        out.append(Image.fromarray(np.clip(o,0,255).astype(np.uint8)))
    return out