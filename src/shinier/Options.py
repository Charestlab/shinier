# Global imports
from typing import Union, Optional, Iterable
from pathlib import Path

class Options:
    """
    Class to hold SHINE processing options.

    Args:
        images_format (str): png, tif, jpg (default = tif)
        masks_format (str): png, tif, jpg (default = tif)
        input_folder (Union[str, Path]): relative or absolute path of the image folder (default = ./INPUT)
        output_folder (Union[str, Path]): relative or absolute path where processed images will be saved (default = ./OUTPUT)
        masks_folder (Union[str, Path]): relative or absolute path of mask (default = ./MAKS)

        iterations (int): number of iterations (default = 1)

        whole_image (int): Default = 1
            1 = whole image (default)
            2 = figure-ground separated (input images as mask(s))
            3 = figure-ground separated (based on mask(s))

        mode (int): Default = 8
            1 = lum_match only
            2 = hist_match only
            3 = sf_match only
            4 = spec_match only
            5 = hist_match & sf_match
            6 = hist_match & spec_match
            7 = sf_match & hist_match
            8 = spec_match & hist_match (default)

        background (int or float): Background lum of mask, or 300=automatic (default)
            (automatically, the luminance that occurs most frequently in the image is used as background lum);
            basically, all regions of that lum are treated as background

        dithering (bool): If True, apply dithering after processing is done and before final conversion to uint8

        rescaling (int): Default = 1. Post-processing applied after sf_match or spec_match only.
            0 = no rescaling
            1 = rescale to min and max of all images (default)
            2 = rescale to average min and max values across all images

        hist_specification (int): Default = 0
            0 = Exact specification without noise (see Coltuc, Bolon & Chassery, 2006)
            1 = Exact specification with noise (legacy code)

        hist_optim (int): Default = 0
            0 = no SSIM optimization
            1 = SSIM optimization (Avanaki, 2009; to change the number if iterations (default = 10) and adjust step size (default = 67), see below)

        iterations (int): Number of iterations for SSIM optimization in hist_optim. Default is 10.
        step_size (int): Step size for SSIM optimization in hist_optim. Default is 67.
        
        seed (int): Optional seed to initialize the PRNG.

    """
    def __init__(
            self,
            images_format: str = 'tif',
            masks_format: str = 'tif',
            input_folder: Union[str, Path] = Path('./../../INPUT'),
            output_folder: Union[str, Path] = Path('./../../OUTPUT'),
            masks_folder: Optional[Union[str, Path]] = Path("./../../MASK") if Path("./../../MASK").is_dir() and any(Path("./../../MASK").iterdir()) else None,

            whole_image: int = 1,
            conserve_memory: bool = True,
            as_gray: bool = False,

            iterations: int = 1,
            mode: int = 8,
            background: Union[int, float] = 300,
            target_lum: Optional[Iterable[Union[int, float]]] = None,
            safe_lum_match: bool = False,
            hist_specification: int = 0,
            dithering: bool = True,
            rescaling: int = 1,
            hist_optim: int = 0,
            iterations: int = 10,
            step_size: int = 67,
            seed: Optional[int] = None,
            legacy_mode: bool = False
    ):
        self.images_format = images_format
        self.input_folder = Path(input_folder).resolve()
        self.output_folder = Path(output_folder).resolve()

        self.masks_format = masks_format if whole_image == 3 else None
        self.masks_folder = Path(masks_folder).resolve() if whole_image == 3 else None

        self.whole_image = whole_image

        self.conserve_memory = conserve_memory
        self.as_gray = as_gray

        self.mode = mode
        self.background = background
        self.target_lum = target_lum
        self.safe_lum_match = safe_lum_match
        self.hist_specification = hist_specification
        self.dithering = dithering
        self.rescaling = 0 if mode==1 else rescaling
        self.hist_optim = hist_optim
        self.iterations = iterations
        self.step_size = step_size


        self.seed = seed
        self.legacy_mode = legacy_mode

        # Override validation and
        if self.legacy_mode != True:
            self._validate_options()
        else:
            self.conserve_memory = False
            self.as_gray = True
            self.dithering = False
            self.hist_specification = 1
            self.safe_lum_match = False

    def __repr__(self):
        """Provides a detailed representation of all options for easy inspection."""
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])

    def _validate_options(self):
        """Validates the options to ensure they are within acceptable ranges."""
        if self.whole_image not in [1, 2, 3]:
            raise ValueError("whole_image must be 1, 2 or 3. See Options")
        if self.mode not in [1, 2, 3, 4, 5, 6, 7, 8]:
            raise ValueError("Invalid mode selected. See Options")
        if self.background not in range(0,256) and self.background != 300:
            raise ValueError("background must be [0, 255] or 300")
        if self.rescaling not in [0, 1, 2]:
            raise ValueError("Rescaling must be 0, 1, or 2. See Options")
        if self.hist_specification not in [0, 1]:
            raise ValueError("hist_specification must be 0 or 1. See Options")
        if self.hist_optim not in [0, 1]:
            raise ValueError("Optim must be 0 or 1. See Options")
        if self.iterations < 1:
            raise ValueError("Iterations must be at least 1. See Options")
        if self.step_size < 1:
            raise ValueError("Step size must be at least 1. See Options")
        if self.rescaling != 0 and self.mode in [1, 2]:
            raise ValueError("Should not apply rescaling after luminance or histogram matching.")
        if not self.input_folder.is_dir():
            raise ValueError(f"{self.input_folder} folder does not exists")
        if not self.output_folder.is_dir():
            raise ValueError(f"{self.output_folder} folder does not exists")
        if self.masks_folder != None:
            if not self.masks_folder.is_dir():
                    raise ValueError(f"{self.masks_folder} folder does not exists")
