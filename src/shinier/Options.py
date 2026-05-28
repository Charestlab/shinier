from __future__ import annotations
from pathlib import Path
from typing import Union, Optional, Literal, Tuple, Any, get_args
import numpy as np
import json
from pydantic import (
    Field,
    ConfigDict,
    conint,
    confloat,
    field_validator,
    model_validator,
    PrivateAttr
)
from pydantic.json_schema import model_json_schema, GenerateJsonSchema, JsonSchemaValue
from shinier.utils import console_log, Bcolors
from shinier.base import InformativeBaseModel
from shinier import REPO_ROOT
ACCEPTED_IMAGE_FORMATS = Literal["png", "tif", "tiff", "jpg", "jpeg", "npy"]
GAMUT_STRATEGY_TYPE = Literal[
    'constrain_dataset_luminance',
    'constrain_dataset_chrominance',
    'constrain_image_chrominance',
    'constrain_image_luminance',
    'clip'
]
OPTION_TYPES = {
    'io':               ['input_folder', 'output_folder'],
    'mask':             ['masks_folder', 'background', 'whole_image'],
    'mode':             ['mode', 'legacy_mode', 'seed', 'iterations'],
    'color':            ['as_gray', 'linear_luminance', 'rec_standard', 'gamut_strategy'],
    'dithering_memory': ['dithering', 'conserve_memory'],
    'luminance':        ['safe_lum_match', 'target_lum'],
    'histogram':        ['hist_optim', 'hist_specification', 'hist_iterations', 'target_hist'],
    'fourier':          ['rescaling', 'target_spectrum', 'fft_padding_mode', 'fft_padding_value'],
    'misc':             ['verbose']
}


class Options(InformativeBaseModel):
    """
    Class to hold SHINIER processing options.

    Subsections
    -----------
    1. INPUT/OUTPUT images folders
        ``input_folder``, ``output_folder``
    2. MASKS and FIGURE-GROUND separation
        ``masks_folder``, ``whole_image``, ``background``
    3. SHINIER MODE
        ``mode``, ``legacy_mode``, ``seed``, ``iterations``
    4. Grayscale / color
        ``as_gray``, ``linear_luminance``, ``rec_standard``, ``gamut_strategy``
    5. Dithering / Memory
        ``dithering``, ``conserve_memory``
    6. LUMINANCE matching
        ``safe_lum_match``, ``target_lum``
    7. HISTOGRAM matching
        ``hist_optim``, ``hist_specification``, ``hist_iterations``, ``target_hist``
    8. FOURIER matching
        ``rescaling``, ``target_spectrum``, ``fft_padding_mode``, ``fft_padding_value``
    9. Misc
        ``verbose``

    Options
    -------
    input_folder : Union[str, Path]
        [1] INPUT/OUTPUT images folders.

        Relative or absolute path of the image folder.
        Default is the package sample INPUT folder.

    output_folder : Union[str, Path]
        [1] INPUT/OUTPUT images folders.

        Relative or absolute path where processed images will be saved.
        Default is the package sample OUTPUT folder.

    masks_folder : Optional[Union[str, Path]]
        [2] MASKS and FIGURE-GROUND separation.

        Relative or absolute path of mask folder.
        Default is None.

    whole_image : Literal[1, 2, 3]
        [2] MASKS and FIGURE-GROUND separation.

        Binary region-of-interest (ROI) masks: analysis runs on selected pixels.
        Default is 1.

        - 1 = No ROI mask: whole images will be analyzed.
        - 2 = ROI masks: analysis run on pixels != ``background`` pixel value.
        - 3 = ROI masks: masks loaded from the MASK folder and analysis run on pixels >= 127.

    background : Union[int, float]
        [2] MASKS and FIGURE-GROUND separation.

        Background grayscale intensity of mask, or 300 = automatic.
        Default is 300.

        By default (300), the most frequent luminance intensity in the image is used as the background value;
        i.e., all regions of that luminance intensity are treated as background.

    mode : Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]
        [3] SHINIER MODE.

        Image processing treatment.
        Default is 2.

        - 1 = lum_match only.
        - 2 = hist_match only (default).
        - 3 = sf_match only.
        - 4 = spec_match only.
        - 5 = hist_match and sf_match.
        - 6 = hist_match and spec_match.
        - 7 = sf_match and hist_match.
        - 8 = spec_match and hist_match.
        - 9 = only dithering.

    legacy_mode : Optional[bool]
        [3] SHINIER MODE.

        Enables backward compatibility with older versions while retaining recent optimizations.
        Default is False.

        Important: legacy_mode affects more than the explicit option overrides listed below.
        It also enables MATLAB-compatibility behavior in several processing steps
        (for example MATLAB-style rounding and grayscale conversion paths), so outputs may differ
        even when two runs appear to share the same visible option values.

        True reproduces the behavior of previous releases by setting:

        - ``conserve_memory`` = ``False``
        - ``as_gray`` = ``1``
        - ``dithering`` = ``0``
        - ``hist_specification`` = ``1``
        - ``safe_lum_match`` = ``False``

        False means no legacy settings are forced and all options follow their current defaults.

    seed : Optional[int]
        [3] SHINIER MODE.

        Seed to initialize the PRNG.
        Default is None.

        Used for the Noisy bit dithering and ``hist_specification`` (with ``hybrid`` or ``noise`` tie-breaking strategies).
        If ``None``, ``int(time.time())`` will be used.

    iterations : int
        [3] SHINIER MODE.

        Number of iterations for composite modes.
        Default is 5.

        For these modes, histogram specification and Fourier amplitude specification affect each other.
        Multiple iterations allow a high degree of joint matching.

        This method of iterating was developed so that it recalculates the respective target at each iteration
        (i.e., no target hist/spectrum).

    as_gray : bool
        [4] Grayscale / color.

        Conversion into grayscale images.
        Default is 0 (False).

        - True = Convert into grayscale images.

                - When ``linear_luminance`` is ``False``:
                    computes non-linear grayscale images by applying the perceptual
                    luma weights from the specified ``rec_standard``.
                    
                - When ``linear_luminance`` is ``True``:
                    computes linear grayscale images by averaging the RGB channels
                    (simple mean(RGB)).

        - False = No conversion applied.

    linear_luminance : bool
        [4] Grayscale / color.

        Are pixel values linearly related to luminance?
        Default is False.

        - True: no conversion mode.

                - Assumes input images are linear RGB or grayscale.
                - All transformations are applied independently to each channel.
                - No color-space conversion is performed.

        - False: conversion to CIE xyY (recommended and default).

                - Assumes input images are gamma-encoded (e.g., sRGB).
                - Images are converted to the CIE xyY color space:
                    sRGB -> linRGB -> XYZ -> xyY
                - Transformations are applied only to the luminance channel (Y),
                    while chromatic channels (x, y) remain unchanged.
                - The modified image is then reconstructed via:
                    xyY -> XYZ -> linRGB -> sRGB
                - This mode preserves color gamuts and is highly recommended for operations
                    on linear-to-luminance values like fourier matching and luminance matching.

    rec_standard : Literal[1, 2, 3]
        [4] Grayscale / color.

        Specifies the Rec. color standard used for RGB <-> XYZ conversion.
        Default is 2.

        - 1 = Rec.601 (SDTV, legacy systems).
        - 2 = Rec.709 (HDTV, sRGB default). SHINIER assumes display-referred Rec. 709 with sRGB-like transfer.
        - 3 = Rec.2020 (UHDTV, wide-gamut HDR).

        
    gamut_strategy : Literal['constrain_dataset_luminance', 'constrain_dataset_chrominance', 'constrain_image_chrominance', 'constrain_image_luminance', 'clip']
        [4] Grayscale / color.

        Strategy to deal with out-of-gamut problem.
        Requires ``linear_luminance=False``.
        Default is ``constrain_image_chrominance``.

        Global constraints (applies the same transform to the whole dataset): best for dataset consistency.

            - ``constrain_dataset_luminance``: Scales the luminance of ALL images down so the most saturated pixel fits.
                Preserves hue and saturation; compresses contrast/luminance.

            - ``constrain_dataset_chrominance``: Scales the saturation of ALL images down so the brightest pixel fits.
                Preserves contrast/luminance; compresses saturation.

        Local repairs (applies a single transform to all pixels of a given image): best to maximize image contrast.

            - ``constrain_image_chrominance``: Darkens all pixels so that there are no out-of-gamut pixels.

            - ``constrain_image_luminance``: Desaturates all pixels so that there are not out-of-gamut pixels.

            - ``clip``: Default color conversion behavior, numpy ``safe_mode`` (not recommended unless you know what are you doing).

    dithering : Literal[0, 1, 2]
        [5] Dithering / Memory.

        Dithering applied before final conversion to uint8 to mitigate precision loss 
        from float64 processing and perceptually extend effective bit depth.
        
        Default is 0.

        - 0 = No dithering.
        - 1 = Noisy bit dithering (Allard R. and Faubert J., 2008).
        - 2 = Floyd-Steinberg dithering (Floyd R.W. and Steinberg L., 1976).

    conserve_memory : Optional[bool]
        [5] Dithering / Memory.

        Controls how images are loaded and stored in memory during processing.
        Default is True.

        True minimizes memory usage by keeping only one image in memory at a time and using a temporary directory
        to save the images. If the ``input_data`` is a list of NumPy arrays, images are first saved as ``.npy`` in a
        temporary directory, and they are loaded in memory one at a time upon request.

        False increases memory usage substantially by loading all images into memory at once,
        but may improve processing speed.

    safe_lum_match : bool
        [6] LUMINANCE matching.

        Adjusting the mean and standard deviation to keep all luminance values [0, 255].
        Default is True.

        True = No values will be clipped, but the resulting targets may differ from the requested values.
        False = Values will be clipped, but the resulting targets will stay the same.

    target_lum : Tuple[Optional[float], Optional[float]]
        [6] LUMINANCE matching.

        Target luminance statistics as ``(mean, std)`` for luminance matching.
        Default is ``(0, 0)``.

        Constraints:

        - ``mean`` must be in ``[0, 255]`` or ``None``.
        - ``std`` must be ``>= 0`` or ``None``.

        Semantics:

        - ``0`` uses the dataset average for that statistic.
        - ``None`` leaves that statistic unchanged per image.

        Mean and standard deviation can be controlled independently by setting
        either value to ``None``.

    hist_optim : bool
        [7] HISTOGRAM matching.

        Optimization of the histogram-matched images with structural similarity index measure (Avanaki, 2009).
        Default is False.

                - True = SSIM optimization (Avanaki, 2009).

                        - Following Avanaki's experimental results, no tie-breaking strategy is applied when
                            optimizing SSIM except for the very last iteration where the hybrid strategy is used
                            (see ``hist_specification``).

                        - To change the number of iterations (default = 5) and adjust step size
                            (default = 35), see below.

                - False = No SSIM optimization.

    hist_specification : Literal[1, 2, 3, 4]
        [7] HISTOGRAM matching.

        Determines the algorithm used to break ties (isoluminance) when matching the histogram.
        Default is 4.

        - 1 = ``Noise``: Exact specification with noise (legacy code).
            Add small uniform noise to break ties (fast; non-deterministic unless seed set).

        - 2 = ``Moving-average``: Coltuc Bolon and Chassery (2006) tie-breaking strategy with moving-average filters.
            Kernels defined in the paper sorted lexicographically for deterministic local ordering.

        - 3 = ``Gaussian``: Coltuc tie-breaking strategy with gaussian filters.
            Adaptive amount of gaussian filters used (min 5, max 7; deterministic local ordering).

        - 4 = ``Hybrid``: Coltuc tie-breaking strategy with gaussian filters and noise fallback if isoluminant pixels persist.
            ``Gaussian`` (deterministic) + ``Noise fallback`` (stochastic, if needed), best compromise.

        Set to ``None`` if ``hist_optim`` is ``True``. See ``hist_optim`` for more info.

    hist_iterations : int
        [7] HISTOGRAM matching.

        Number of iterations for SSIM optimization in hist_optim.
        Default is 10.

    target_hist : Optional[Union[np.ndarray, Path, Literal['equal']]]
        [7] HISTOGRAM matching.

        Target histogram, image path, ``'equal'``, or ``None``.
        Default is ``None``.

        Accepted inputs:

            - ``np.ndarray`` (histogram array):
                    - Can contain histogram counts (int) or weights (float).
                        Histograms are normalized internally before use.
                    - Shape must be ``(256,)`` or ``(256, 1)`` for single-channel processing,
                        or ``(256, C)`` for multi-channel processing
                        (e.g., RGB with ``linear_luminance=True``, ``as_gray=False``).
                    - See ``imhist`` in ``utils.py`` to compute a target histogram from an image.

            - ``Path`` (input image file):
                    - Image is processed using the same pipeline as the dataset to compute the target histogram.
                    - Spatial dimensions must match the processed images.

            - ``'equal'``:
                    - Uses a flat histogram, i.e., histogram equalization.

            - ``None``:
                    - Uses the average histogram of all input images.

        Used in all modes involving histogram matching (modes 2, 5, 6, 7, and 8).

    rescaling : Literal[0, 1, 2, 3]
        [8] FOURIER matching.

        Post-processing applied after sf_match or spec_match only.
        Default is 2.

        - 0 = no rescaling.
        - 1 = Rescaling each image so that it stretches to [0, 1] (its own min -> 0, max -> 1).
        - 2 = Rescaling absolute max/min (shared 0-1 range).
        - 3 = Rescaling average max/min.

        Not allowed for modes 1 and 2.

    target_spectrum : Optional[Union[np.ndarray, Path]]
        [8] FOURIER matching.

        Target Fourier magnitude spectrum, image path, or ``None``.
        Default is ``None``.

        Accepted inputs:

            - ``np.ndarray`` (magnitude spectrum array, dtype must be float):
                    - Spatial shape must match the processed images: ``(H, W)``, or ``(H, W, C)``
                        for multi-channel processing
                        (e.g., RGB with ``linear_luminance=True``, ``as_gray=False``).
                    - See ``image_spectrum`` in ``utils.py`` to compute a target spectrum from an image.

            - ``Path`` (input image file):
                    - Image is processed using the same pipeline as the dataset to compute the target spectrum.
                    - Spatial dimensions must match the processed images.

            - ``None``:
                    - Uses the average spectrum of all input images.

        Used in all modes involving Fourier matching (modes 3, 4, 5, 6, 7, and 8).

    fft_padding_mode : Literal[0, 1, 2, 3]
        [8] FOURIER matching.

        Optional spatial padding before FFT computation.
        Default is 0.

        - 0 = No padding (disabled).
        - 1 = ``Reflect`` : mirror image values without repeating the edge pixel.
        - 2 = ``Symmetric`` : mirror image values including the edge pixel.
        - 3 = ``Constant`` : pad with a constant spatial-domain intensity.

        Padding is applied before FFT and cropped after inverse FFT reconstruction.

    fft_padding_value : Union[int, Literal[300]]
        [8] FOURIER matching.

        Constant padding intensity in [0, 255] when ``fft_padding_mode=3``.
        ``300`` means: use the mean intensity of the current normalized image.
        Default is ``300``.

        If ``300``, the mean intensity of the current normalized image is used.
        Used only when ``fft_padding_mode=3``.

    verbose : Literal[-1, 0, 1, 2, 3], optional
        [9] Misc.

        Controls verbosity levels.
        Default is 0.

        - -1 = Quiet mode.
        - 0 = Progress bar with ETA.
        - 1 = Basic progress steps (no progress bar).
        - 2 = Additional info about image and channels being processed are printed (no progress bar).
        - 3 = Debug mode for developers (no progress bar).

    """
    model_config = ConfigDict(
        validate_assignment=True,  # Validate every time object updated
        extra="forbid",  # Does not allow unknown attributes
        arbitrary_types_allowed=True,  # Allow non-pydantic types (e.g. np.ndarray)
    )

    # --- I/O ---
    input_folder: Optional[Path] = Field(default=REPO_ROOT / "data/INPUT")
    output_folder: Path = Field(default=REPO_ROOT / "data/OUTPUT")

    # --- Masks ---
    masks_folder: Optional[Path] = Field(default=None)
    whole_image: Literal[1, 2, 3] = 1
    background: Union[conint(ge=0, le=255), Literal[300]] = 300

    # --- Mode ---
    mode: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9] = 2
    seed: Optional[int] = None
    legacy_mode: bool = False
    iterations: conint(ge=1) = 5

    # --- Color ---
    as_gray: bool = False
    linear_luminance: bool = False
    rec_standard: Literal[1, 2, 3] = 2
    gamut_strategy: GAMUT_STRATEGY_TYPE = Field(default='constrain_image_chrominance')

    # --- Dithering / Memory ---
    dithering: Literal[0, 1, 2] = 0
    conserve_memory: bool = True

    # --- Luminance ---
    safe_lum_match: bool = True
    target_lum: Tuple[Optional[confloat(ge=0, le=255)], Optional[confloat(ge=0)]] = (0, 0)

    # --- Histogram ---
    hist_optim: bool = False
    hist_specification: Optional[Literal[1, 2, 3, 4]] = 4
    hist_iterations: conint(ge=1) = 10
    target_hist: Optional[Union[np.ndarray, Path, Literal["equal", "unit_test"]]] = Field(default=None)

    # --- Fourier ---
    rescaling: Optional[Literal[0, 1, 2, 3]] = 2
    target_spectrum: Optional[Union[np.ndarray, Path, Literal["unit_test"]]] = Field(default=None)
    fft_padding_mode: Literal[0, 1, 2, 3] = 0
    fft_padding_value: Union[int, Literal[300]] = 300

    # --- Misc ---
    verbose: Literal[-1, 0, 1, 2, 3] = 0

    # --- Private attributes ---
    _is_moving_target: bool = PrivateAttr(default=True)

    # ================================================================================================
    # FIELD-LEVEL VALIDATIONS
    # ================================================================================================
    @field_validator("input_folder", "output_folder", "masks_folder")
    @classmethod
    def validate_existing_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None:
            v = v.resolve()
            if not v.exists():
                raise ValueError(f"Folder does not exist: {v}")
        return v

    @field_validator("target_hist")
    @classmethod
    def validate_target_hist(cls, v):
        """Validate that target_hist is 'equal', an array of correct shape, or a valid image path."""
        if v is None or (isinstance(v, str) and v in ["equal", 'unit_test']):
            return v
        if isinstance(v, (str, Path)):
            v = Path(v).resolve()
            if not v.exists():
                raise ValueError(f"target_hist image does not exist: {v}")
            if not v.is_file():
                raise ValueError(f"target_hist path must point to a file: {v}")
            if v.suffix.lower().lstrip(".") not in get_args(ACCEPTED_IMAGE_FORMATS):
                raise ValueError(
                    f"target_hist image must use one of {get_args(ACCEPTED_IMAGE_FORMATS)}. "
                    f"Got: {v.suffix}"
                )
            return v
        if not isinstance(v, np.ndarray):
            raise TypeError("target_hist must be a numpy.ndarray, an image path, or 'equal'.")
        if v.ndim not in (1, 2):
            raise ValueError("target_hist must be 1D (gray) or 2D (color).")
        if v.ndim == 1 and v.size != 256:
            raise ValueError("For grayscale, target_hist must have 256 bins.")
        return v

    @field_validator("target_spectrum")
    @classmethod
    def validate_target_spectrum(cls, v):
        """Ensure target_spectrum is a float np.ndarray, a valid image path, or 'unit_test'."""
        if v is None or (isinstance(v, str) and v in ['unit_test']):
            return v
        if isinstance(v, (str, Path)):
            v = Path(v).resolve()
            if not v.exists():
                raise ValueError(f"target_spectrum image does not exist: {v}")
            if not v.is_file():
                raise ValueError(f"target_spectrum path must point to a file: {v}")
            if v.suffix.lower().lstrip(".") not in get_args(ACCEPTED_IMAGE_FORMATS):
                raise ValueError(
                    f"target_spectrum image must use one of {get_args(ACCEPTED_IMAGE_FORMATS)}. "
                    f"Got: {v.suffix}"
                )
            return v
        if not isinstance(v, np.ndarray):
            raise TypeError("target_spectrum must be a numpy.ndarray or an image path.")
        if not np.issubdtype(v.dtype, np.floating):
            raise TypeError("target_spectrum dtype must be float.")
        return v

    @field_validator("target_lum")
    @classmethod
    def validate_target_lum(cls, v):
        """Reject target_lum values that would not adjust any luminance statistic."""
        if v[0] is None and v[1] is None:
            raise ValueError("target_lum=(None, None) does not adjust mean or contrast.")
        return v

    @field_validator("fft_padding_mode", mode="before")
    @classmethod
    def validate_fft_padding_mode(cls, v):
        """Normalize FFT padding mode to numbered options used in the public API."""
        if v is None:
            return 0
        if isinstance(v, bool):
            raise TypeError("fft_padding_mode must be an integer in {0, 1, 2, 3}.")
        if isinstance(v, int) and 0 <= v <= 3:
            return v
        raise TypeError("fft_padding_mode must be 0, 1, 2, or 3.")

    @field_validator("fft_padding_value", mode="before")
    @classmethod
    def validate_fft_padding_value(cls, v):
        """Normalize FFT padding value so 300 is the explicit mean-intensity sentinel."""
        if not isinstance(v, int):
            raise TypeError("fft_padding_value must be an integer in [0, 255] or 300.")
        if v != 300 and not (0 <= v <= 255):
            raise ValueError("fft_padding_value must be in [0, 255] or 300.")
        return v

    # ================================================================================================
    # CROSS-FIELD LOGIC VALIDATION
    # ================================================================================================
    @model_validator(mode="after")
    def cross_checks(self) -> "Options":
        """Enforce consistency between interdependent fields."""

        # Rescaling not valid for luminance/histogram modes → Overwrite and warn
        if self.mode in (1, 2) and self.rescaling not in (None, 0):
            object.__setattr__(self, "rescaling", 0)
            console_log(msg=f"Rescaling not valid for luminance/histogram modes. rescaling -> 0", color=Bcolors.WARNING, verbose=self.verbose > 0)

        # Mode 9: must have dithering != 0 → raise ValueError
        if self.mode == 9 and self.dithering == 0:
            raise ValueError("Mode 9 requires dithering 1 or 2 (not 0).")

        # target_hist should match expected images size under as_gray and linear_luminance
        if self.target_hist is not None and not isinstance(self.target_hist, (str, Path)):
            if (not self.linear_luminance or self.as_gray) and self.target_hist.size != 256:
                raise ValueError(f"target_hist must be (256, ) or (256, 1) when linear_luminance is False or as_gray is True. Current target_hist shape = {self.target_hist.shape}")

        # target_spectrum should match expected images size under as_gray and linear_luminance
        if self.target_spectrum is not None:
            if not self.linear_luminance or self.as_gray:
                if not isinstance(self.target_spectrum, (str, Path)) and self.target_spectrum.squeeze().ndim != 2:
                    raise ValueError(f"target_spectrum must be (W, H,) or (W, H, 1) when linear_luminance is False or as_gray is True. Current target_spectrum shape = {self.target_spectrum.shape}")

        # hist_specification ignored if hist_optim = True
        if self.hist_optim:
            object.__setattr__(self, "hist_specification", None)
            console_log(msg=f"hist_specification ignored if hist_optim = True. hist_specification -> None", color=Bcolors.WARNING, verbose=self.verbose > 0)

        # fft_padding_value is only meaningful for mode 3 (constant padding)
        if self.fft_padding_mode != 3 and self.fft_padding_value != 300:
            object.__setattr__(self, "fft_padding_value", 300)
            console_log(msg="fft_padding_value ignored unless fft_padding_mode = 3. fft_padding_value -> 300", color=Bcolors.WARNING, verbose=self.verbose > 0)

        # whole_image == 3 → requires mask folder & format
        if self.whole_image == 3:
            if self.masks_folder is None:
                raise ValueError("whole_image=3 requires a valid masks_folder.")

        # iterations > 1 only valid for composite modes (5–8) -> overwrite and warn
        if self.iterations > 1 and self.mode not in (5, 6, 7, 8):
            object.__setattr__(self, "iterations", 1)
            console_log(msg="Iterations > 1 ignored outside composite modes (5–8). iterations → 1", color=Bcolors.WARNING, verbose=self.verbose > 0)

        # Gamut strategy constraints are only supported in xyY conversion mode (linear_luminance=False)
        if not self.as_gray:
            if self.gamut_strategy != 'clip' and self.linear_luminance:
                object.__setattr__(self, "gamut_strategy", 'clip')
                if self.verbose > 0:
                    console_log(msg="Gamut constraints require xyY conversion mode (linear_luminance=False). gamut_strategy -> 'clip'", color=Bcolors.WARNING)

        if self.gamut_strategy != 'clip' and self.as_gray:
            object.__setattr__(self, "gamut_strategy", 'clip')
            if self.verbose > 0:
                console_log(msg="Gamut strategy ignored for grayscale images. gamut_strategy -> 'clip'",
                            color=Bcolors.WARNING)

        # Legacy overrides
        if self.legacy_mode:
            object.__setattr__(self, "conserve_memory", False)
            object.__setattr__(self, "as_gray", True)
            object.__setattr__(self, "linear_luminance", False)
            object.__setattr__(self, "rec_standard", 1)
            object.__setattr__(self, "dithering", 0)
            object.__setattr__(self, "hist_specification", 1)
            object.__setattr__(self, "safe_lum_match", False)

        return self

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @classmethod
    def model_json_schema(
        cls,
        *,
        by_alias: bool = True,
        ref_template: str = "#/$defs/{model}",
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        **kwargs: Any,
    ) -> JsonSchemaValue:
        """Custom safe JSON schema that skips unsupported types."""

        class SafeJsonSchema(schema_generator):
            """Gracefully replace np.ndarray and Path in schema generation."""

            def is_instance_schema(self, schema) -> JsonSchemaValue:
                typ = schema.get("cls")
                if typ is np.ndarray:
                    return {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Placeholder for numpy.ndarray",
                    }
                if typ is Path:
                    return {
                        "type": "string",
                        "format": "path",
                        "description": "Filesystem path",
                    }
                # Default fallback
                return super().handle_invalid_for_json_schema(
                    schema, f"Unsupported type {typ}"
                )

        # Call the internal helper manually (no recursion!)
        return model_json_schema(
            cls,
            ref_template=ref_template,
            schema_generator=SafeJsonSchema,
            by_alias=by_alias,
            **kwargs,
        )

    def export_schema(self, file_path: Path) -> None:
        out = Path(file_path)
        indent = 2
        ensure_ascii = True

        # Create parent dir if does not exist
        out.parent.mkdir(parents=True, exist_ok=True)

        # Get model schema and write it in file_path
        json_schema = self.model_json_schema()
        with out.open("w", encoding="utf-8") as f:
            json.dump(json_schema, f, indent=indent, ensure_ascii=ensure_ascii)

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.model_dump().items())

    def _assumptions_warning(self):
        msg = None
        if self.as_gray > 0:
            if self.mode == 1:
                msg = ('[warning] Luminance matching assumes linear relation '
                       'to luminance, which is not true for sRGB.')
            elif self.mode in [2, 5, 6, 7, 8]:
                msg = ("[warning] `hist_match` operates directly on intensity values "
                       "and does not assume linear luminance scaling.")
        else:
            if self.mode in [2, 5, 6, 7, 8]:
                msg = ('[warning] `hist_match` applied per-channel may cause '
                       'out-of-gamut colors; use joint RGB histograms for consistency.')
        if msg:
            console_log(msg, indent_level=0, color=Bcolors.WARNING, verbose=self.verbose >= 1)
