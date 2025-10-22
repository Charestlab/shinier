# Global imports
from shinier.utils import console_log, Bcolors
from typing import Union, Optional, Iterable, List, Literal
from pathlib import Path
import numpy as np


class Options:
    """
    Class to hold SHINIER processing options.

    Args:
    ----------------------------------------------INPUT/OUTPUT images folders-------------------------------------------------
        images_format (str): png, tif, jpg (default = png)

        input_folder (Union[str, Path]): relative or absolute path of the image folder (default = ./INPUT)

        output_folder (Union[str, Path]): relative or absolute path where processed images will be saved (default = ./OUTPUT)

    -------------------------------------------MASKS and FIGURE-GROUND separation----------------------------------------------
        masks_format (str): png, tif, jpg (default = png)
        masks_folder (Union[str, Path]): relative or absolute path of mask (default = ./MASKS)

        whole_image (Literal[1-3]): Binary ROI masks: Analysis run on selected pixels (default = 1)
            1 = No ROI mask: Whole images will be analyzed
            2 = ROI masks: Analysis run on pixels != `background` pixel value
            3 = ROI masks: Masks loaded from the `MASK` folder and analysis run on pixels >= 127

        background (Union[int, float]): Background grayscale intensity of mask, or 300 = automatic (default = 300)
            (By default (300), the most frequent luminance intensity in the image is used as the background value);
            i.e., all regions of that luminance intensity are treated as background

    ------------------------------------------SHINIER MODE, COLORS, RAM management---------------------------------------------
        mode (Literal[1-9]): (default = 8)
            1 = lum_match only
            2 = hist_match only
            3 = sf_match only
            4 = spec_match only
            5 = hist_match & sf_match
            6 = hist_match & spec_match
            7 = sf_match & hist_match
            8 = spec_match & hist_match (default)
            9 = only dithering

        as_gray (Literal[0-4]): Defines how the images are converted into grayscale before being converted to uint8 (default = 0).
            0 = No conversion applied
            1 = Equal weighted sum of R, G and B pixels is applied. (Y' = 1/3 R' + 1/3 B' + 1/3 G').
            2 = Rec.ITU-R 601 is used (legacy mode; see Matlab).    (Y' = 0.299 R' + 0.587 G' + 0.114 B') (Standard-Definition monitors)
            3 = Rec.ITU-R 709 is used.                              (Y' = 0.2126 R' + 0.7152 G' + 0.0722 B') (High-Definition monitors)
            4 = Rec.ITU-R 2020 is used.                             (Y' = 0.2627 R' + 0.6780 G' + 0.0593 B') (Ultra-High-Definition monitors)

               >The prime notation (') indicates that the RGB values have been gamma-corrected, meaning they have undergone a
                non-linear transformation to match human visual perception and display characteristics, ensuring faithful color
                reproduction on modern displays.

        dithering (Literal[0-2]): Dithering applied before final conversion to uint8 (default = 1).
            0 = No dithering
            1 = Noisy bit dithering (Allard R. & Faubert J., 2008)
            2 = Floyd-Steinberg dithering (Floyd R.W. & Steinberg L., 1976)

        conserve_memory (Optional[bool]): Controls how images are loaded and stored in memory during processing (default = True).
            True = Minimizes memory usage by keeping only one image in memory at a time and using a temporary directory to save the images.
                If the `input_data` is a list of NumPy arrays images are first saved as .npy in a temporary directory, and they are loaded
                in memory one at a time upon request.
            False = Increases memory usage substantially by loading all images into memory at once, but may improve processing speed.

        seed (Optional[Int]): Seed to initialize the PRNG (default = None).
            Used for the 'Noisy bit dithering' and hist_specification (with "hybrid" or "noise" tie-breaking strategies).
            If 'None', int(time.time()) will be used.

        legacy_mode (Optional[bool]): Enables backward compatibility with older versions while retaining recent optimizations (default = False).
            True = reproduces the behavior of previous releases by setting:
                - `conserve_memory = False`
                - `as_gray = 2`
                - `dithering = 0`
                - `hist_specification = 1`
                - `safe_lum_match = False`
            False = no legacy settings are forced and all options follow their current defaults.

        iterations (int): Number of iteration for composites mode (default = 2).
            For these modes, histogram specification and Fourier amplitude specification affect each other.
            Multiple iterations will allows a high degree a joint matching.

                >This method of iterating was develop so that it recalculates the respective target at each iteration (i.e., no target hist/spectrum).

    --------------------------------------------------HISTOGRAM matching--------------------------------------------------------
        hist_optim (bool): Optimization of the histogram-matched images with structural similarity index measure (Avanaki, 2009) (default = False)
            True = SSIM optimization (Avanaki, 2009)
                    > Following Avanaki's experimental results, no tie-breaking strategy is applied when optimizing SSIM except for the very last iteration where the "hybrid" hist_specification is used.
                    > To change the number if iterations (default = 5) and adjust step size (default = 35), see below
            False = No SSIM optimization

        hist_specification (Literal[1-4]): Determines the algorithm used to break the ties (isoluminance) when matching the histogram (default = 4).
            >> Set to None if hist_optim is True. See hist_optim for more info.
            1 = 'Noise': Exact specification with noise (legacy code)
                    > Add small uniform noise to break ties (fast; non-deterministic unless seed set).
            2 = 'Moving-average': Coltuc Bolon & Chassery (2006) tie-breaking strategy with moving-average filters.
                    > Kernels defined in the paper sorted lexicographically for deterministic local ordering.
            3 = 'Gaussian': Coltuc's tie-breaking strategy with gaussian filters.
                    > Adaptive amount of gaussian filters used (min 5, max 7; deterministic local ordering).
            4 = 'Hybrid': Coltuc's tie-breaking strategy with gaussian filters, then noise if isoluminant pixels persist.
                    > 'Gaussian' (deterministic) + 'Noise' (stochastic; if needed) - best compromise.

        hist_iterations (int): Number of iterations for SSIM optimization in hist_optim (default is 10).

        step_size (int): Step size for SSIM optimization in hist_optim (default is 35).
                    > This initial measure is adjusted during the optimization process using Avanaki's (2009) theoretical bounds.

        target_hist (Optional[np.ndarray, Literal['equal']]): Target histogram counts (int) or weights (float) to use for histogram or fourier matching (default is None).
            Should be a numpy array of shape (256,) for 8-bit images, or a string 'equal' for histogram equalization.
            If 'None', the target histogram is the average histogram of all the input images.
            E.g.,
                from shinier.utils import imhist
                target_hist = imhist(im)

    --------------------------------------------------LUMINANCE matching------------------------------------------------------
        safe_lum_match (bool): Adjusting the mean and standard deviation to keep all luminance values [0, 255] (default = False).
            True = No values will be clipped, but the resulting targets may differ from the requested values.
            False = Values will be clipped, but the resulting targets will stay the same.

        target_lum (Optional[Iterable[Union[int, float]]]): Pair (mean, std) of target luminance for luminance matching (default = (0, 0)).
            The mean must be in [0, 255], and the standard deviation must be ≥ 0.
            If (0, 0), the mean and std will be the average mean and average std of the images.
            Only for mode 1.

        rgb_weights (Literal[1-4]): RGB values are converted to luminance using a weighted sum for lum_match computation (default = 3).
            1 = Equal weighted sum of R, G and B pixels is applied. (Y' = 1/3 R' + 1/3 B' + 1/3 G').
            2 = Rec.ITU-R 601 is used (legacy mode ; see Matlab).   (Y' = 0.299 R' + 0.587 G' + 0.114 B')    (Standard-Definition monitors)
            3 = Rec.ITU-R 709 is used.                              (Y' = 0.2126 R' + 0.7152 G' + 0.0722 B') (High-Definition monitors)
            4 = Rec.ITU-R 2020 is used.                             (Y' = 0.2627 R' + 0.6780 G' + 0.0593 B') (Ultra-High-Definition monitors)

    --------------------------------------------------FOURIER matching--------------------------------------------------------
        rescaling (Literal[0-3]): Post-processing applied after sf_match or spec_match only (default = 2).
            0 = no rescaling
            1 = Rescaling each image so that it stretches to [0, 1] (its own min→0, max→1).
            2 = Rescaling absolute max/min (shared 0–1 range).
            3 = Rescaling average max/min.

        target_spectrum: Optional[np.ndarray[float]]: Target magnitude spectrum (default = None).
            Same size as the images of float values.
            If 'None', the target magnitude spectrum is the average spectrum of all the input images.
            Only for mode 3 and 4.
            E.g.,
                from shinier.utils import cart2pol
                fftim = np.fft.fftshift(np.fft.fft2(im))
                rho, theta = cart2pol(np.real(fftim), np.imag(fftim))
                target_spectrum = rho

    --------------------------------------------------EXTRA--------------------------------------------------------
        verbose (Literal[-1, 0, 1, 2, 3]): Controls verbosity levels (default = 0).
            -1 = Quiet mode
            0 = Progress bar with ETA
            1 = Basic progress steps (no progress bar)
            2 = Additional info about image and channels being processed are printed (no progress bar)
            3 = Debug mode for developers (no progress bar)

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
            legacy_mode: Optional[bool] = False,

            safe_lum_match: bool = False,
            target_lum: Optional[Iterable[Union[int, float]]] = (0, 0),
            rgb_weights: Literal[1, 2, 3, 4] = 3,

            hist_specification: Literal[1, 2, 3, 4] = 4,
            hist_optim: bool = False,
            hist_iterations: int = 10,
            step_size: int = 35,
            target_hist: Optional[Union[np.ndarray, Literal['equal']]] = None,

            rescaling: Optional[Literal[0, 1, 2, 3]] = 2,
            target_spectrum: Optional[np.ndarray] = None,
            iterations: int = 2,

            verbose: Literal[-1, 0, 1, 2] = 0  # -1: Nothing is printed (used for unit tests); 0: Minimal processing steps are printed; 1: Additional info about image and channels being processed are printed; 2: Additional info about the results of internal tests are printed.
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

        self.conserve_memory = conserve_memory if mode in [5, 6, 7, 8] else False
        self.seed = seed

        self.safe_lum_match = safe_lum_match
        self.target_lum = target_lum
        self.rgb_weights = rgb_weights

        self.hist_specification = None if hist_optim else hist_specification
        self.hist_optim = hist_optim
        self.hist_iterations = hist_iterations
        self.step_size = step_size
        self.target_hist = target_hist

        self.rescaling = 0 if self.mode in [1, 2] else rescaling if rescaling is not None else 2
        self.target_spectrum = target_spectrum
        self.iterations = iterations if mode in [5,6,7,8] else 1

        self.verbose = verbose

        # Override validation
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
            raise ValueError(f"{self.input_folder} folder does not exist")
        if not self.output_folder.is_dir():
            raise ValueError(f"{self.output_folder} folder does not exist")
        if self.masks_folder is not None:
            if not self.masks_folder.is_dir():
                raise ValueError(f"{self.masks_folder} folder does not exist")
        if self.images_format not in ['png', 'tif', 'tiff', 'jpg', 'jpeg']:
            raise ValueError("images format must be either 'png', 'tif', 'tiff', 'jpg' or 'jpeg'")
        if self.masks_format not in ['png', 'tif', 'tiff', 'jpg', 'jpeg'] and self.whole_image == 3:
            raise ValueError("masks format must be either 'png', 'tif', 'tiff', 'jpg' or 'jpeg' if whole_image == 3")
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
        if not isinstance(self.safe_lum_match, bool):
            raise TypeError("safe_lum_match must be a boolean value.")
        if not (isinstance(self.target_lum, Iterable) and all([isinstance(item, (float, int)) for item in self.target_lum]) and len(self.target_lum) == 2):
            raise ValueError("target_lum should be an iterable of two numbers")
        if not (0 <= self.target_lum[0] <= 255):
            raise ValueError(f"Mean luminance is {self.target_lum[0]} but should be between 0 and 255")
        if self.target_lum[1] < 0:
            raise ValueError(f"Standard deviation is {self.target_lum[1]} but should be greater than or equal to 0")

        if not self.hist_optim and self.hist_specification not in [1, 2, 3, 4]:
            raise ValueError("hist_specification must be 1, 2, 3 or 4. See Options")
        if not isinstance(self.hist_optim, bool):
            raise TypeError("hist_optim must be a boolean value (True or False).")
        if self.hist_iterations < 1:
            raise ValueError("hist_iterations must be at least 1. See Options")
        if self.iterations < 1:
            raise ValueError("Iterations must be at least 1. See Options")
        if self.step_size < 1:
            raise ValueError("Step size must be at least 1. See Options")
        if self.target_hist is not None:
            if (not isinstance(self.target_hist, (np.ndarray, str))) or (isinstance(self.target_hist, str) and self.target_hist != 'equal'):
                raise TypeError("target_hist must be either a numpy.ndarray or a string = 'equal' for histogram equalization.")
            if isinstance(self.target_hist, np.ndarray):
                if self.as_gray:
                    if self.target_hist.squeeze().ndim != 1:
                        raise ValueError("For grayscale images (as_gray is 1, 2, 3, 4), target_hist must be 1D (shape (256,)).")
                    if np.prod(self.target_hist.shape) not in [256]:
                        raise ValueError("target_hist must have 256 for 8 bits images.")
                else:
                    if self.target_hist.ndim != 2:
                        raise ValueError("For color images (as_gray = 0), target_hist must be 2D (shape (256, 3)).")
                    if self.target_hist.shape[0] not in [256] or self.target_hist.shape[1] != 3:
                        raise ValueError("target_hist must have shape (256, 3) for RGB images.")

            # if not np.issubdtype(self.target_hist.dtype, np.integer):
            #     raise TypeError("target_hist must contain integer values (pixel counts per bin).")
        if self.rgb_weights not in [1, 2, 3, 4]:
            raise TypeError("rgb_weights must be an int equal to 1, 2, 3 or 4.")

        if self.rescaling != 0 and self.mode in [1, 2]:  # TODO: Shouldn't we prevent rescaling anytime lum and hist match are applied?
            raise ValueError("Should not apply rescaling after luminance or histogram matching.")
        if self.rescaling not in [0, 1, 2, 3]:
            raise ValueError("Rescaling must be 0, 1, 2 or 3. See Options")
        if self.target_spectrum is not None :
            if not isinstance(self.target_spectrum, np.ndarray):
                raise TypeError('The target spectrum must be a numpy array of floats.')
            if not np.issubdtype(self.target_spectrum.dtype, np.floating):
                raise TypeError('The target spectrum must be a numpy array of floats.')
        if self.mode == 9 and self.dithering == 0:
            raise ValueError("The dithering option cannot be 0 for mode 9. Should be either 1 or 2.")

        if self.verbose not in [-1, 0, 1, 2]:
            raise ValueError('Verbose should be -1, 0, 1 or 2. See docstrings for definition.')

    def _assumptions_warning(self):
        msg = None
        if self.as_gray > 0:
            if self.mode in [1]:
                msg = '[warning to user] Luminance matching assumes that the input images are linearly related to luminance, which is not the case for most sRGB images.'
            if self.mode in [2, 5, 6, 7, 8]:
                msg = "[warning to user] `hist_match` operates directly on intensity values and does not assume linear luminance scaling."
        else:
            if self.mode in [2, 5, 6, 7, 8]:
                msg = '\n'.join([
                    "[warning to user] By default, `hist_match` is applied independently to each color channel.",
                    'This may produce inaccurate color relationships or out-of-gamut results.',
                    "If joint color consistency is required, consider using histogram matching of joint RGB distributions",
                    "or other color-aware distribution matching methods."])
        if msg is not None:
            console_log(msg=msg, indent_level=0, color=Bcolors.WARNING, verbose=self.verbose >= 1)

