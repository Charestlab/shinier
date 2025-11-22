from __future__ import annotations
from pathlib import Path
from typing import Union, Optional, Literal, Tuple, Any
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
OPTION_TYPES = {
    'io':               ['input_folder', 'output_folder'],
    'mask':             ['masks_folder', 'background', 'whole_image'],
    'mode':             ['mode', 'legacy_mode', 'seed', 'iterations'],
    'color':            ['as_gray', 'linear_luminance', 'rec_standard'],
    'dithering_memory': ['dithering', 'conserve_memory'],
    'luminance':        ['safe_lum_match', 'target_lum'],
    'histogram':        ['hist_optim', 'hist_specification', 'hist_iterations', 'target_hist'],
    'fourier':          ['rescaling', 'target_spectrum'],
    'misc':             ['verbose']
}


class Options(InformativeBaseModel):
    """
    Class to hold SHINIER processing options.

    Args:
    ------------------------------------------INPUT/OUTPUT images folders------------------------------------------
        input_folder (Union[str, Path]): relative or absolute path of the image folder (default = ./INPUT)

        output_folder (Union[str, Path]): relative or absolute path where processed images will be saved (default = ./OUTPUT)

    ------------------------------------------MASKS and FIGURE-GROUND separation------------------------------------------
        masks_folder (Union[str, Path]): relative or absolute path of mask (default = ./MASKS)

        whole_image (Literal[1-3]): Binary ROI masks: Analysis run on selected pixels (default = 1)
            1 = No ROI mask: Whole images will be analyzed
            2 = ROI masks: Analysis run on pixels != `background` pixel value
            3 = ROI masks: Masks loaded from the `MASK` folder and analysis run on pixels >= 127

        background (Union[int, float]): Background grayscale intensity of mask, or 300 = automatic (default = 300)
            (By default (300), the most frequent luminance intensity in the image is used as the background value);
            i.e., all regions of that luminance intensity are treated as background

    --------------------------------------------------   SHINIER MODE  --------------------------------------------------
        mode (Literal[1-9]): Image processing treatment (default = 8)
            1 = lum_match only
            2 = hist_match only (default)
            3 = sf_match only
            4 = spec_match only
            5 = hist_match & sf_match
            6 = hist_match & spec_match
            7 = sf_match & hist_match
            8 = spec_match & hist_match
            9 = only dithering

        legacy_mode (Optional[bool]): Enables backward compatibility with older versions while retaining recent optimizations (default = False).
            True = reproduces the behavior of previous releases by setting:
                - `conserve_memory = False`
                - `as_gray = 1`
                - `dithering = 0`
                - `hist_specification = 1`
                - `safe_lum_match = False`
            False = no legacy settings are forced and all options follow their current defaults.

        seed (Optional[Int]): Seed to initialize the PRNG (default = None).
            Used for the 'Noisy bit dithering' and hist_specification (with "hybrid" or "noise" tie-breaking strategies).
            If 'None', int(time.time()) will be used.

        iterations (int): Number of iteration for composites mode (default = 2).
            For these modes, histogram specification and Fourier amplitude specification affect each other.
            Multiple iterations will allows a high degree a joint matching.

                >This method of iterating was develop so that it recalculates the respective target at each iteration (i.e., no target hist/spectrum).

    --------------------------------------------------Grayscale / color------------------------------------------------------
        as_gray (bool): Conversion into grayscale images (default = 0).
            False = No conversion applied.
            True = Convert into grayscale images.
                - Using `rec_standard` if `linear_luminance` is False
                - Using simple mean(RGB) if `linear_luminance` is True

        linear_luminance (bool): Are pixel values linearly related to luminance? (default = False).
            True: "no conversion" mode
                - Assumes input images are linear RGB or grayscale.
                - All transformations are applied independently to each channel.
                - No color-space conversion is performed.

            False: "conversion to xyY" [recommended and default]
                - Assumes input images are gamma-encoded (e.g., sRGB).
                - Images are converted to the CIE xyY color space:
                      sRGB → linRGB → XYZ → xyY
                - Transformations are applied only to the luminance channel (Y),
                  while chromatic channels (x, y) remain unchanged.
                - The modified image is then reconstructed via:
                      xyY → XYZ → linRGB → sRGB
                - This mode preserves color gamuts and is highly recommended
                  for operations on linear-to-luminance values like fourier matching and luminance matchning.

        rec_standard (Literal[1, 2, 3]): Specifies the Rec. color standard used for RGB ↔ XYZ conversion (default = 2).
            1 = Rec.601 (SDTV, legacy systems)
            2 = Rec.709 (HDTV, sRGB default). Shinier assumes display-referred Rec. 709 with sRGB-like transfer.
            3 = Rec.2020 (UHDTV, wide-gamut HDR)

    --------------------------------------------------Dithering / Memory------------------------------------------------------
        dithering (Literal[0-2]): Dithering applied before final conversion to uint8 (default = 0).
            0 = No dithering
            1 = Noisy bit dithering (Allard R. & Faubert J., 2008)
            2 = Floyd-Steinberg dithering (Floyd R.W. & Steinberg L., 1976)

        conserve_memory (Optional[bool]): Controls how images are loaded and stored in memory during processing (default = True).
            True = Minimizes memory usage by keeping only one image in memory at a time and using a temporary directory to save the images.
                If the `input_data` is a list of NumPy arrays images are first saved as .npy in a temporary directory, and they are loaded
                in memory one at a time upon request.
            False = Increases memory usage substantially by loading all images into memory at once, but may improve processing speed.

    --------------------------------------------------LUMINANCE matching------------------------------------------------------
        safe_lum_match (bool): Adjusting the mean and standard deviation to keep all luminance values [0, 255] (default = False).
            True = No values will be clipped, but the resulting targets may differ from the requested values.
            False = Values will be clipped, but the resulting targets will stay the same.

        target_lum (Optional[Iterable[Union[int, float]]]): Pair (mean, std) of target luminance for luminance matching (default = (0, 0)).
            The mean must be in [0, 255], and the standard deviation must be ≥ 0.
            If (0, 0), the mean and std will be the average mean and average std of the images.
            Only for mode 1.

    --------------------------------------------------HISTOGRAM matching--------------------------------------------------------
        hist_optim (bool): Optimization of the histogram-matched images with structural similarity index measure (Avanaki, 2009) (default = False)
            True = SSIM optimization (Avanaki, 2009)
                    >> Following Avanaki's experimental results, no tie-breaking strategy is applied when optimizing SSIM except for the very last
                       iteration where the "hybrid" strategy is used (see hist_specification).
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

        target_hist (Optional[np.ndarray, Literal['equal']]): Target histogram counts (int) or weights (float) to use for histogram or fourier matching (default is None).
            Should be a numpy array of shape (256,) for 8-bit images, or a string 'equal' for histogram equalization.
            If 'None', the target histogram is the average histogram of all the input images.
            E.g.,
                from shinier.utils import imhist
                target_hist = imhist(im)

    --------------------------------------------------FOURIER matching--------------------------------------------------------
        rescaling (Literal[0-3]): Post-processing applied after sf_match or spec_match only (default = 2).
            0 = no rescaling
            1 = Rescaling each image so that it stretches to [0, 1] (its own min→0, max→1).
            2 = Rescaling absolute max/min (shared 0–1 range).
            3 = Rescaling average max/min.
            > Not allowed for modes 1 and 2.

        target_spectrum: Optional[np.ndarray[float]]: Target magnitude spectrum (default = None).
            Same size as the images of float values.
            If 'None', the target magnitude spectrum is the average spectrum of all the input images.
            Only for mode 3 and 4.
            E.g.,
                from shinier.utils import cart2pol
                fftim = np.fft.fftshift(np.fft.fft2(im))
                rho, theta = cart2pol(np.real(fftim), np.imag(fftim))
                target_spectrum = rho

    --------------------------------------------------Misc--------------------------------------------------------
        verbose (Literal[-1, 0, 1, 2, 3]): Controls verbosity levels (default = 0).
            -1 = Quiet mode
            0 = Progress bar with ETA
            1 = Basic progress steps (no progress bar)
            2 = Additional info about image and channels being processed are printed (no progress bar)
            3 = Debug mode for developers (no progress bar)

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

    # --- Dithering / Memory ---
    dithering: Literal[0, 1, 2] = 0
    conserve_memory: bool = True

    # --- Luminance ---
    safe_lum_match: bool = False
    target_lum: Tuple[conint(ge=0, le=255), confloat(ge=0)] = (0, 0)

    # --- Histogram ---
    hist_optim: bool = False
    hist_specification: Optional[Literal[1, 2, 3, 4]] = 4
    hist_iterations: conint(ge=1) = 10
    target_hist: Optional[Union[np.ndarray, Literal["equal", "unit_test"]]] = Field(default=None)

    # --- Fourier ---
    rescaling: Optional[Literal[0, 1, 2, 3]] = 2
    target_spectrum: Optional[Union[np.ndarray, Literal["unit_test"]]] = Field(default=None)

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
        """Validate that target_hist is 'equal' or an array of correct shape."""
        if v is None or (isinstance(v, str) and v in ["equal", 'unit_test']):
            return v
        if not isinstance(v, np.ndarray):
            raise TypeError("target_hist must be a numpy.ndarray or 'equal'.")
        if v.ndim not in (1, 2):
            raise ValueError("target_hist must be 1D (gray) or 2D (color).")
        if v.ndim == 1 and v.size != 256:
            raise ValueError("For grayscale, target_hist must have 256 bins.")
        return v

    @field_validator("target_spectrum")
    @classmethod
    def validate_target_spectrum(cls, v):
        """Ensure target_spectrum is float np.ndarray."""
        if v is None or (isinstance(v, str) and v in ['unit_test']):
            return v
        if not isinstance(v, np.ndarray):
            raise TypeError("target_spectrum must be a numpy.ndarray.")
        if not np.issubdtype(v.dtype, np.floating):
            raise TypeError("target_spectrum dtype must be float.")
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
        if self.target_hist is not None and not isinstance(self.target_hist, str):
            if (not self.linear_luminance or self.as_gray) and self.target_hist.size != 256:
                raise ValueError(f"target_hist must be (256, ) or (256, 1) when linear_luminance is False or as_gray is True. Current target_hist shape = {self.target_hist.shape}")

        # target_spectrum should match expected images size under as_gray and linear_luminance
        if self.target_spectrum is not None:
            if not self.linear_luminance or self.as_gray:
                if not isinstance(self.target_spectrum, str) and self.target_spectrum.squeeze().ndim != 2:
                    raise ValueError(f"target_spectrum must be (W, H,) or (W, H, 1) when linear_luminance is False or as_gray is True. Current target_spectrum shape = {self.target_spectrum.shape}")

        # hist_specification ignored if hist_optim = True
        if self.hist_optim:
            object.__setattr__(self, "hist_specification", None)
            console_log(msg=f"hist_specification ignored if hist_optim = True. hist_specification -> None", color=Bcolors.WARNING, verbose=self.verbose > 0)

        # whole_image == 3 → requires mask folder & format
        if self.whole_image == 3:
            if self.masks_folder is None:
                raise ValueError("whole_image=3 requires a valid masks_folder.")

        # iterations > 1 only valid for composite modes (5–8) -> overwrite and warn
        if self.iterations > 1 and self.mode not in (5, 6, 7, 8):
            object.__setattr__(self, "iterations", 1)
            console_log(msg="Iterations > 1 ignored outside composite modes (5–8). iterations → 1", color=Bcolors.WARNING, verbose=self.verbose > 0)

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
