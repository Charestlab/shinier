# Global imports
from typing import Union, Optional, Iterable, List, Literal
from pathlib import Path
import numpy as np


class Options:
    """
    Class to hold SHINIER processing options.

    Args:
    ----------------------------------------------INPUT/OUTPUT images folders-------------------------------------------------
        images_format (str): png, tif, tiff, jpg, jpeg (default = tif)
        input_folder (Union[str, Path]): relative or absolute path of the image folder (default = ./INPUT)
        output_folder (Union[str, Path]): relative or absolute path where processed images will be saved (default = ./OUTPUT)

    -------------------------------------------MASKS and FIGURE-GROUND separation----------------------------------------------
        masks_format (str): png, tif, tiff, jpg, jpeg (default = tif)
        masks_folder (Union[str, Path]): relative or absolute path of mask (default = ./MASKS)

        whole_image (Literal): Default = 1
            1 = whole image (default)
            2 = figure-ground separated (input images as mask(s))
            3 = figure-ground separated (based on mask(s))

        background (int or float): Background lum of mask, or 300=automatic (default)
            (automatically, the luminance that occurs most frequently in the image is used as background lum);
            basically, all regions of that lum are treated as background

    ------------------------------------------SHINIER MODE, COLORS, RAM management---------------------------------------------
        mode (Literal): Default = 8
            1 = lum_match only
            2 = hist_match only
            3 = sf_match only
            4 = spec_match only
            5 = hist_match & sf_match
            6 = hist_match & spec_match
            7 = sf_match & hist_match
            8 = spec_match & hist_match (default)
            9 = only dithering

        as_gray (Optional[int]): Images are converted into grayscale then uint8. Default is no conversion (default = 0).
            0 = No conversion applied
            1 = An equal weighted sum of red, green and blue pixels is applied.
            2 = (legacy mode) Rec.ITU-R 601 is used (see Matlab). Y′ = 0.299 R′ + 0.587 G′ + 0.114 B′ (Standard-Definition monitors)
            3 = Rec.ITU-R 709 is used. Y′ = 0.2126 R′ + 0.7152 G′ + 0.0722 B′ (High-Definition monitors)
            4 = Rec.ITU-R 2020 is used. Y′ = 0.2627 R′ + 0.6780 G′ + 0.0593 B′ (Ultra-High-Definition monitors)

        dithering (Literal): Default = 1, dithering before final conversion to uint8
            0 = no dithering
            1 = noisy bit dithering (Allard R. & Faubert J. (2008))
            2 = Floyd-Steinberg dithering (Floyd R.W. & Steinberg L., 1976)

        conserve_memory (Optional[bool]): If True (default), uses a temporary directory to store images
            and keeps only one image in memory at a time. If True and input_data is a list of NumPy arrays,
            images are first saved as .npy in a temporary directory, and they are loaded in memory one at a time upon request.

        seed (Optional[Int]): Optional seed to initialize the PRNG. Random is used for noisy bit dithering and for exact histogram
            specification with noise.

        legacy_mode (bool): If True, ensures compatibility with older versions and workflows, preserving previous functionalities
            while integrating new optimizations. (conserve_memory = False, as_gray = 2, dithering = 0, hist_specification = 1,
            safe_lum_match = False)
        
        iterations (int): Default = 2, number of iteration for composites mode. For these modes, histogram specification and Fourier 
            amplitude specification affect each other. Multiple iterations will allows a high degree a joint matching.
            !! The method was develop so that it recalculates the respective target at each iteration (i.e., no target hist/spectrum).

    --------------------------------------------------HISTOGRAM matching--------------------------------------------------------
        hist_specification (Literal): Default = 0
            0 = Exact specification without noise (see Coltuc, Bolon & Chassery, 2006)
            1 = Exact specification with noise (legacy code)

        hist_optim (Literal): Default = 0
            0 = no SSIM optimization
            1 = SSIM optimization (Avanaki, 2009; to change the number if iterations (default = 10) and adjust step size (default = 35), see below)

        hist_iterations (int): Number of iterations for SSIM optimization in hist_optim. Default is 10.
        step_size (int): Step size for SSIM optimization in hist_optim. Default is 35. (Avanaki (2009) uses 67)

        target_hist (Optional[np.ndarray]): Target histogram counts (int) or weights (float) to use for histogram or fourier matching. Should be a
            numpy array of shape (256,) or (65536,) for 8-bit or 16-bit images, or as required by the processing function.
            Default is None. Only for mode 2.
            E.g.,
                from shinier.utils import imhist
                target_hist = imhist(im)

    --------------------------------------------------LUMINANCE matching------------------------------------------------------
        safe_lum_match (bool): Default = False.
            If True, adjusts the target mean and standard deviation to keep all luminance values within [0, 255];
            the resulting targets may differ from the requested values.

        target_lum (Optional[Iterable[Union[int, float]]]): Default = (0, 0)
            Pair (mean, std) of target luminance for luminance matching. The mean must be in [0, 255], and the standard deviation must be ≥ 0.
            The mean must be in [0, 255], and the standard deviation must be ≥ 0.
            Only for mode 1.

    --------------------------------------------------FOURIER matching--------------------------------------------------------
        rescaling (Literal): Default = 2. Post-processing applied after sf_match or spec_match only.
            0 = no rescaling
            1 : Rescaling each image so that it stretches to [0, 1]
            2 : Rescaling absolute max/min (Default)
            3 : Rescaling average max/min

        target_spectrum: Optional[np.ndarray[float]]: Target magnitude spectrum.
            Same size as the images of float values. If None, the target magnitude
            spectrum is the average spectrum of all the input images.
            Only for mode 3 and 4.
            E.g.,
                from shinier.utils import cart2pol
                fftim = np.fft.fftshift(np.fft.fft2(im))
                rho, theta = cart2pol(np.real(fftim), np.imag(fftim))
                target_spectrum = rho

    """
    def __init__(
            self,

            images_format: str = 'png',
            input_folder: Union[str, Path] = Path('./../INPUT'),
            output_folder: Union[str, Path] = Path('./../OUTPUT'),

            masks_format: str = 'png',
            masks_folder: Optional[Union[str, Path]] = Path("./../MASK") if Path("./../MASK").is_dir() and any(Path("./../MASK").iterdir()) else None,
            whole_image: Literal[1, 2, 3] = 1,
            background: Union[int, float] = 300,

            mode: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9] = 8,
            as_gray: Literal[0, 1, 2, 3, 4] = 0,
            dithering: Literal[0, 1, 2] = 1,
            conserve_memory: bool = True,
            seed: Optional[int] = None,
            legacy_mode: bool = False,

            safe_lum_match: bool = False,
            target_lum: Optional[Iterable[Union[int, float]]] = (0, 0),

            hist_specification: Literal[0, 1] = 0,
            hist_optim: Literal[0, 1] = 0,
            hist_iterations: int = 5,
            step_size: int = 34,
            target_hist: Optional[np.ndarray] = None,

            rescaling: Optional[Literal[0, 1, 2, 3]] = 2,
            target_spectrum: Optional[np.ndarray] = None,
            iterations: int = 2

    ):
        self.images_format = images_format
        self.input_folder = Path(input_folder).resolve()
        self.output_folder = Path(output_folder).resolve()

        self.masks_format = masks_format if whole_image == 3 else None
        self.masks_folder = Path(masks_folder).resolve() if whole_image == 3 else None
        self.whole_image = whole_image
        self.background = background

        self.legacy_mode = legacy_mode
        self.mode = mode
        self.as_gray = 2 if self.legacy_mode else as_gray
        self.dithering = dithering

        self.conserve_memory = conserve_memory if mode in [5,6,7,8] else False
        self.seed = seed

        self.safe_lum_match = safe_lum_match
        self.target_lum = target_lum

        self.hist_specification = hist_specification
        self.hist_optim = hist_optim
        self.hist_iterations = hist_iterations
        self.step_size = step_size
        self.target_hist = target_hist

        self.rescaling = 0 if self.mode in [1, 2] else rescaling if rescaling is not None else 2
        self.target_spectrum = target_spectrum
        self.iterations = iterations if mode in [5,6,7,8] else 1

        # Override validation and
        if not self.legacy_mode:
            self._validate_options()
        else:
            self.conserve_memory = False
            self.as_gray = 2
            self.dithering = 0
            self.hist_specification = 1
            self.safe_lum_match = False

    def __repr__(self):
        """Provides a detailed representation of all options for easy inspection."""
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])

    def _validate_options(self):
        """Validates the options to ensure they are within acceptable ranges."""
        if not self.input_folder.is_dir():
            raise ValueError(f"{self.input_folder} folder does not exists")
        if not self.output_folder.is_dir():
            raise ValueError(f"{self.output_folder} folder does not exists")
        if self.masks_folder is not None:
            if not self.masks_folder.is_dir():
                raise ValueError(f"{self.masks_folder} folder does not exists")
        if self.images_format not in ['png', 'tif', 'tiff', 'jpg', 'jpeg']:
            raise ValueError("images format must be either 'png', 'tif', 'tiff', 'jpg' or 'jpeg'")
        if self.masks_format not in ['png', 'tif', 'tiff', 'jpg', 'jpeg'] and self.whole_image == 3 :
            raise ValueError("masks format muse be either 'png', 'tif', 'tiff', 'jpg' or 'jpeg' if whole_image == 3")
        if self.whole_image not in [1, 2, 3]:
            raise ValueError("whole_image must be 1, 2 or 3. See Options")
        if self.background not in range(0,256) and self.background != 300:
            raise ValueError("background must be [0, 255] or 300")

        if self.mode not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            raise ValueError("Invalid mode selected. See Options")
        if self.as_gray not in [0, 1, 2, 3, 4]:
            raise TypeError("as_gray must be an int equal to 0, 1, 2, 3 or 4.")
        if self.dithering not in [0, 1, 2]:
            raise ValueError("dithering must be an int equal to 0, 1 or 2.")
        if not isinstance(self.conserve_memory, bool):
            raise TypeError("conserve_memory must be a boolean value.")
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("seed must be an integer value or None.")
        if not isinstance(self.legacy_mode, bool):
            raise TypeError("legacy_mode must be a boolean value.")

        if not isinstance(self.legacy_mode, bool):
            raise TypeError("legacy_mode must be a boolean value.")
        if not isinstance(self.safe_lum_match, bool):
            raise TypeError("safe_lum_match must be a boolean value.")
        if not (isinstance(self.target_lum, Iterable) and all([isinstance(item, (float, int)) for item in self.target_lum]) and len(self.target_lum) == 2):
            raise ValueError("target_lum should be an iterable of two numbers")
        if not (self.target_lum[0] >= 0 and self.target_lum[0] <= 255):
            raise ValueError(f"Mean luminance is {self.target_lum[0]} but should be between 0 and 255")
        if self.target_lum[1] < 0:
            raise ValueError(f"Standard deviation is {self.target_lum[1]} but should be greater than or equal to 0")

        if self.hist_specification not in [0, 1]:
            raise ValueError("hist_specification must be 0 or 1. See Options")
        if self.hist_optim not in [0, 1]:
            raise ValueError("Optim must be 0 or 1. See Options")
        if self.hist_iterations < 1:
            raise ValueError("hist_iterations must be at least 1. See Options")
        if self.iterations < 1:
            raise ValueError("Iterations must be at least 1. See Options")
        if self.step_size < 1:
            raise ValueError("Step size must be at least 1. See Options")
        if self.target_hist is not None:
            if not isinstance(self.target_hist, np.ndarray):
                raise TypeError("target_hist must be a numpy.ndarray.")
            if self.as_gray:
                if self.target_hist.squeeze().ndim != 1:
                    raise ValueError("For grayscale images (as_gray is 1, 2, 3, 4), target_hist must be 1D (shape (256,) or (65536,)).")
                if np.prod(self.target_hist.shape) not in [256, 65536]:
                    raise ValueError("target_hist must have 256 or 65536 values (for 8 or 16 bits).")
            else:
                if self.target_hist.ndim != 2:
                    raise ValueError("For color images (as_gray = 0), target_hist must be 2D (shape (256, 3) or (65536, 3)).")
                if self.target_hist.shape[0] not in [256, 65536] or self.target_hist.shape[1] != 3:
                    raise ValueError("target_hist must have shape (256, 3) or (65536, 3) for RGB images.")
            # if not np.issubdtype(self.target_hist.dtype, np.integer):
            #     raise TypeError("target_hist must contain integer values (pixel counts per bin).")

        if self.rescaling != 0 and self.mode in [1, 2]:  # TODO: Shouldn't we prevent rescaling anytime lum and hist match are applied?
            raise ValueError("Should not apply rescaling after luminance or histogram matching.")
        if self.rescaling not in [0, 1, 2, 3]:
            raise ValueError("Rescaling must be 0, 1, 2 or 3. See Options")
        if self.target_spectrum is not None :
            if not isinstance(self.target_spectrum, np.ndarray):
                raise TypeError('The target spectrum must be a numpy array of np.float64.')
            if not np.issubdtype(self.target_spectrum.dtype, np.floating):
                raise TypeError('The target spectrum must be a numpy array of np.float64.')
        if self.mode == 9 and self.dithering == 0:
            raise ValueError("The dithering option cannot be 0 for mode 9. Should be either 1 or 2.")