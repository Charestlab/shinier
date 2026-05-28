"""
shinier.color.converter
=======================

Core color-space conversion class for Shinier.

Implements reversible pipelines:
    sRGB → linRGB → XYZ → CIELAB
    CIELAB → XYZ → linRGB → sRGB

Validation tests: /tests/validation_tests/Converter_validation_tests.py
    - Results replicate the colour-science package (https://pypi.org/project/colour-science/)
    across all tested colour standards (Rec.601, Rec.709, Rec.2020).

    - Slight deviation observed for Rec.601 arises from our use of a piecewise
    sRGB/Rec.709-style transfer function (γ ≈ 2.2) instead of the historical
    pure power-law (γ = 2.8) applied in older analog-era Rec.601 specifications.
    This choice ensures consistency with MATLAB’s rgb2gray and modern digital
    workflows, where Rec.601 luminance weights are paired with an sRGB-like TRC.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Union, Optional, TYPE_CHECKING, Tuple, ClassVar, Any, get_args
from pydantic import Field, ConfigDict, model_validator
from PIL import Image
from shinier.utils import im3D, console_log, Bcolors
from shinier.base import InformativeBaseModel
from shinier import REPO_ROOT

if TYPE_CHECKING:
    from shinier.ImageListIO import ImageListIO

# D65 white point normalized to Y=1.0 (XYZ)
WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])

# RGB → XYZ conversion matrices for D65
M_RGB2XYZ_601 = np.load(REPO_ROOT / 'color/M_RGB2XYZ_601.npy')
M_RGB2XYZ_709 = np.load(REPO_ROOT / 'color/M_RGB2XYZ_709.npy')
M_RGB2XYZ_2020 = np.load(REPO_ROOT / 'color/M_RGB2XYZ_2020.npy')

COLOR_STANDARDS = {
    "rec601": {"M_RGB2XYZ": M_RGB2XYZ_601, "gamma": 2.2, "white": WHITE_D65},
    "rec709": {"M_RGB2XYZ": M_RGB2XYZ_709, "gamma": 2.4, "white": WHITE_D65},
    "rec2020": {"M_RGB2XYZ": M_RGB2XYZ_2020, "gamma": 2.4, "white": WHITE_D65},
}
REC_STANDARD = Literal["rec601", "rec709", "rec2020"]
RGB_STANDARD = Literal["equal", "rec601", "rec709", "rec2020"]

RGB2GRAY_WEIGHTS = {
    'equal': [1/3, 1/3, 1/3],
    'rec601': M_RGB2XYZ_601[1, :],
    'rec709': M_RGB2XYZ_709[1, :],
    'rec2020': M_RGB2XYZ_2020[1, :],
}
for k, v in RGB2GRAY_WEIGHTS.items():
    RGB2GRAY_WEIGHTS[k] /= np.sum(v)
int2key_mapping = dict(zip(range(1, len(RGB2GRAY_WEIGHTS)+1), RGB2GRAY_WEIGHTS.keys()))
RGB2GRAY_WEIGHTS['int2key'] = int2key_mapping
RGB2GRAY_WEIGHTS['key2int'] = dict(zip(RGB2GRAY_WEIGHTS['int2key'].values(), RGB2GRAY_WEIGHTS['int2key'].keys()))


class ColorConverter(InformativeBaseModel):
    """Encapsulates color-space conversions for Rec.601/709/2020 systems.

    Parameters
    ----------
    rec_standard : Literal["rec601", "rec709", "rec2020"]
        Rec. standard used for transfer functions and RGB/XYZ matrices.

    safe_mode : bool
        If True, clip intermediate RGB values to valid ranges during conversions
        where clipping is appropriate.

    Runtime Attributes
    ------------------
    - ``gamma`` (float): Transfer-function exponent.
    - ``white_point`` (np.ndarray): Reference white point, D65 by default.
    - ``M_RGB2XYZ`` (np.ndarray): Forward RGB-to-XYZ conversion matrix.
    - ``M_XYZ2RGB`` (np.ndarray): Inverse XYZ-to-RGB conversion matrix.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    rec_standard: Literal["rec601", "rec709", "rec2020"] = "rec709"
    gamma: float = 2.4
    safe_mode: bool = True
    white_point: np.ndarray = Field(default_factory=lambda: WHITE_D65.copy())
    M_RGB2XYZ: np.ndarray = Field(default_factory=lambda: M_RGB2XYZ_709.copy())
    M_XYZ2RGB: np.ndarray = Field(default_factory=lambda: np.linalg.inv(M_RGB2XYZ_709))

    # --- Pydantic-setting of attributes as a function of rec_standard ---
    @model_validator(mode="after")
    def apply_standard_config(self) -> "ColorConverter":
        """Apply matrix and transfer-function settings from the selected Rec. standard."""
        cfg = COLOR_STANDARDS[self.rec_standard]
        object.__setattr__(self, "gamma", cfg["gamma"])
        object.__setattr__(self, "white_point", cfg["white"].copy())
        object.__setattr__(self, "M_RGB2XYZ", cfg["M_RGB2XYZ"].copy())
        object.__setattr__(self, "M_XYZ2RGB", np.linalg.inv(cfg["M_RGB2XYZ"]))
        return self

    # ------------------------------------------------------------------
    # sRGB ↔ linRGB
    # ------------------------------------------------------------------
    def sRGB_to_linRGB(self, rgb: np.ndarray) -> np.ndarray:
        """Convert non-linear R'G'B' values to linear RGB.

        Parameters
        ----------
        rgb : np.ndarray
            Gamma-encoded RGB array in [0, 1].

        Returns
        -------
        np.ndarray
            Linear RGB array.

        Notes
        -----
        - ``rec2020`` uses the ITU-R BT.2020 inverse OETF.
        - ``rec709`` uses the IEC 61966-2-1 sRGB transfer function.
        - ``rec601`` uses an sRGB-style piecewise transfer function for
          consistency with MATLAB-like SDR workflows.
        """
        rgb = np.clip(rgb, 0, 1) if self.safe_mode else rgb
        if self.rec_standard == "rec2020":
            alpha, beta = 1.0993, 0.0181
            return np.where(rgb < beta * 4.5, rgb / 4.5, ((rgb + (alpha - 1)) / alpha) ** (1 / 0.45))
        else:
            return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** self.gamma)

    def linRGB_to_sRGB(self, linRGB: np.ndarray) -> np.ndarray:
        """Convert linear RGB values to non-linear R'G'B'.

        Parameters
        ----------
        linRGB : np.ndarray
            Linear RGB array in [0, 1].

        Returns
        -------
        np.ndarray
            Gamma-encoded RGB array.
        """
        linRGB = np.clip(linRGB, 0, 1) if self.safe_mode else linRGB
        if self.rec_standard == "rec2020":
            alpha, beta = 1.0993, 0.0181
            return np.where(linRGB < beta, 4.5 * linRGB, alpha * np.power(linRGB, 0.45) - (alpha - 1))
        else:
            return np.where(linRGB <= 0.0031308, 12.92 * linRGB, 1.055 * np.power(linRGB, 1 / self.gamma) - 0.055)

    # ------------------------------------------------------------------
    # linRGB ↔ XYZ
    # ------------------------------------------------------------------
    def linRGB_to_xyz(self, linRGB: np.ndarray) -> np.ndarray:
        """Convert linear RGB into CIE XYZ.

        Parameters
        ----------
        linRGB : np.ndarray
            Linear RGB array.

        Returns
        -------
        np.ndarray
            CIE XYZ tristimulus values.
        """
        out = np.empty_like(linRGB)
        np.matmul(linRGB.reshape(-1, 3), self.M_RGB2XYZ.T, out=out.reshape(-1, 3))  # linRGB @ self.M_RGB2XYZ.T
        return out

    def xyz_to_linRGB(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ into linear RGB."""
        out = np.empty_like(xyz)
        np.matmul(xyz.reshape(-1, 3), self.M_XYZ2RGB.T, out=out.reshape(-1, 3))  # xyz @ self.M_XYZ2RGB.T
        return np.clip(out, 0, 1, out=out) if self.safe_mode else out

    # ------------------------------------------------------------------
    # XYZ ↔ xyY
    # ------------------------------------------------------------------
    @staticmethod
    def xyz_to_xyY(xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ into CIE xyY.

        The x and y channels encode chromaticity; Y stores luminance.
        """
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        denom = X + Y + Z
        denom_safe = np.where(denom == 0, 1.0, denom)
        x, y = X / denom_safe, Y / denom_safe
        return np.stack([x, y, Y], axis=-1)

    @staticmethod
    def xyY_to_xyz(xyY: np.ndarray, safe_mode: bool = True) -> np.ndarray:
        """Convert CIE xyY into CIE XYZ.

        Parameters
        ----------
        xyY : np.ndarray
            xyY array with channels ``x``, ``y``, and ``Y``.

        safe_mode : bool
            If True, replaces near-zero ``y`` values with 1.0 to avoid numerical
            explosions. If False, preserves tiny ``y`` values so gamut-control
            code can detect out-of-gamut reconstructions.

        Returns
        -------
        np.ndarray
            CIE XYZ tristimulus values.
        """
        x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
        if safe_mode:
            # Safe Mode: Standard conversion
            # If y ~ 0 (invalid), assume y=1 (Neutral/Luminance only).
            # This prevents X/Z explosion for noisy black pixels.
            y_safe = np.where(y <= 1e-9, 1.0, y)
        else:
            # Raw Mode: Gamut Control
            # If y ~ 0, we WANT explosion to detect the gamut violation.
            y_safe = np.where(y <= 1e-9, 1e-9, y)
        X = x * Y / y_safe
        Z = (1 - x - y) * Y / y_safe
        return np.stack([X, Y, Z], axis=-1)

    # ------------------------------------------------------------------
    # XYZ ↔ Lab
    # ------------------------------------------------------------------
    @staticmethod
    def _f_lab(t: np.ndarray) -> np.ndarray:
        """Apply the CIE Lab forward piecewise nonlinearity."""
        epsilon, kappa = 216 / 24389, 24389 / 27
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16) / 116)

    @staticmethod
    def _f_lab_inv(t: np.ndarray) -> np.ndarray:
        """Apply the inverse CIE Lab piecewise nonlinearity."""
        epsilon, kappa = 216 / 24389, 24389 / 27
        return np.where(t > (kappa * epsilon + 16) / 116, t**3, (116 * t - 16) / kappa)

    def xyz_to_lab(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ values to CIE Lab coordinates."""
        xyz_n = xyz / self.white_point
        f = self._f_lab(xyz_n)
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1)

    def lab_to_xyz(self, lab: np.ndarray) -> np.ndarray:
        """Convert CIE Lab values to CIE XYZ coordinates."""
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16) / 116
        fx, fz = fy + a / 500, fy - b / 200
        xyz = np.stack([fx, fy, fz], axis=-1)
        xyz = self._f_lab_inv(xyz)
        return xyz * self.white_point

    # ------------------------------------------------------------------
    # Full pipelines (unchanged)
    # ------------------------------------------------------------------
    def sRGB_to_xyz(self, rgb: np.ndarray) -> np.ndarray:
        """Convert gamma-encoded sRGB values directly to CIE XYZ."""
        return self.linRGB_to_xyz(self.sRGB_to_linRGB(rgb))

    def xyz_to_sRGB(self, xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ values directly to gamma-encoded sRGB."""
        return self.linRGB_to_sRGB(self.xyz_to_linRGB(xyz))

    def sRGB_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert gamma-encoded sRGB values directly to CIE Lab."""
        return self.xyz_to_lab(self.sRGB_to_xyz(rgb))

    def lab_to_sRGB(self, lab: np.ndarray) -> np.ndarray:
        """Convert CIE Lab values directly to gamma-encoded sRGB."""
        return self.xyz_to_sRGB(self.lab_to_xyz(lab))

    def sRGB_to_xyY(self, rgb: np.ndarray) -> np.ndarray:
        """Convert gamma-encoded sRGB values directly to CIE xyY."""
        return self.xyz_to_xyY(self.sRGB_to_xyz(rgb))

    def xyY_to_sRGB(self, xyY: np.ndarray) -> np.ndarray:
        """Convert CIE xyY values directly to gamma-encoded sRGB."""
        return self.xyz_to_sRGB(self.xyY_to_xyz(xyY))


class ColorTreatment(ColorConverter):
    """Apply forward and backward color treatment around SHINIER processing.

    The forward treatment converts input images into the space used by the
    processing pipeline. In the default color-preserving path, sRGB images are
    converted to CIE xyY, the luminance channel is stored in the main buffer,
    and chromaticity channels are stored in an auxiliary buffer. The backward
    treatment reconstructs display-ready sRGB images after processing.

    Runtime Attributes
    ------------------
    - ``Y_desaturation_threshold`` (ClassVar[float]): Low-luminance threshold
      used when preventive chroma desaturation is enabled.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )
    Y_desaturation_threshold: ClassVar[float] = 0.01

    @staticmethod
    def forward_color_treatment(
            rec_standard: REC_STANDARD,
            input_images: ImageListIO,
            output_images: ImageListIO,
            linear_luminance: bool,
            as_gray: bool,
            output_other: Optional[ImageListIO] = None,
            conversion_type: Literal['sRGB_to_xyY', 'sRGB_to_lab'] = 'sRGB_to_xyY',
            desaturate_chroma_on_low_luminance: bool = False,
            legacy_mode: bool = False,
            verbose: bool = False) -> Tuple[ImageListIO, Optional[ImageListIO]]:
        """Apply the forward color-treatment step to an image collection.

        This static method performs a forward pass based on the specified color treatment, grayscale
        conversion, and color space transformation. Depending on the input parameters, the images
        may undergo various treatments such as conversion to grayscale, transformation into a different
        color space (sRGB to xyY or sRGB to Lab), or extracting specific channels for processing.

        Parameters
        ----------
        rec_standard : REC_STANDARD
            Reference color standard for image processing.

        input_images : ImageListIO
            Input image collection to process.

        output_images : ImageListIO
            Buffer receiving the main processed channel or channels.

        linear_luminance : bool
            If True, skips perceptual color-space conversion and processes
            channels directly.

        as_gray : bool
            If True, output is grayscale. If False, color channels are preserved
            through an auxiliary buffer when needed.

        output_other : Optional[ImageListIO]
            Secondary buffer receiving auxiliary chromatic channels.

        conversion_type : Literal["sRGB_to_xyY", "sRGB_to_lab"]
            Forward color conversion to apply.

        desaturate_chroma_on_low_luminance : bool
            If True, desaturates chroma for very low-luminance pixels to prevent
            chromatic noise from being inflated by luminance manipulation.

        legacy_mode : bool
            If True, uses MATLAB-compatible grayscale conversion behavior.

        verbose : bool
            If True, prints processing messages.

        Returns
        -------
        Tuple[ImageListIO, Optional[ImageListIO]]
            Main processed buffer and optional auxiliary chromatic buffer.

        Raises
        ------
        ValueError
            If color treatment requires ``output_other`` but it is not provided.
        """
        if as_gray == 0 and desaturate_chroma_on_low_luminance:
            from shinier.color.GamutControl import GamutControl  # local import avoids circularity
            gamut_control = GamutControl(
                strategy='clip',
                rec_standard=rec_standard,
                color_space='xyY',
                low_Y_desaturate=True,
                low_Y_threshold=ColorTreatment.Y_desaturation_threshold,
                low_Y_fade_width=0.0,
                log_low_Y_chroma_loss=verbose,
            )

        if not linear_luminance and not as_gray and output_other is None:
            raise ValueError("output_other cannot be None when linear_luminance is False and as_gray is false")

        converter = ColorConverter(rec_standard=rec_standard)

        # Promote grayscale (H, W) → (H, W, 3)
        if input_images.n_dims < 3:
            for idx, image in enumerate(input_images):
                output_images[idx] = gray2rgb(image)

        # --- CASE 1: No color treatment ----------------------------------------
        if linear_luminance:
            # Linear-per-channel mode (no perceptual conversion)
            if as_gray:
                # Convert to grayscale using simple mean
                for idx, image in enumerate(input_images):
                    output_images[idx] = rgb2gray(image, conversion_type="equal", matlab_601=legacy_mode)
            else:
                for idx, image in enumerate(input_images):
                    if np.issubdtype(image.dtype, np.uint8):
                        output_images[idx] = image.astype(float)
            return output_images, None

        # --- CASE 2: Color treatment branch -------------------------------------
        elif not linear_luminance:
            for idx, image in enumerate(output_images):
                if conversion_type == "sRGB_to_xyY":
                    # Convert from sRGB → xyY (internally handles gamma decoding)
                    srgb_before = image / 255.0
                    _image = converter.sRGB_to_xyY(srgb_before)
                    Y_orig = _image[:, :, 2]
                    xy_before = _image[:, :, :2]

                    # Extract luminance (Y) channel — for processing
                    output_images[idx] = _image[:, :, 2] * 255

                    # Optionally store x and y for later reconstruction
                    if as_gray == 0:
                        output_other[idx] = _image[:, :, :2]

                    # Optionally desaturate x and y based on original Y
                    if as_gray == 0 and desaturate_chroma_on_low_luminance:
                        _, xy_after = gamut_control.apply_low_Y_desaturation(
                            Y=Y_orig,
                            other=output_other[idx],
                            idx=idx,
                            verbose=verbose,
                        )
                        output_other[idx] = xy_after

                elif conversion_type == "sRGB_to_lab":
                    # Convert from sRGB → Lab (internally handles gamma decoding)
                    _image = converter.sRGB_to_lab(image / 255)

                    # Extract luminance (Y) channel — for processing
                    output_images[idx] = _image[:, :, 0]/100 * 255

                    # Optionally store a and b for later reconstruction
                    if as_gray == 0:
                        output_other[idx] = _image[:, :, 1:]
                else:
                    raise ValueError(f"Unknown conversion type `{conversion_type}`")

            return output_images, output_other
        else:
            raise ValueError(f"Unknown linear luminance `{linear_luminance}`")

    @staticmethod
    def backward_color_treatment(
            rec_standard: REC_STANDARD,
            input_images: ImageListIO,
            output_images: ImageListIO,
            linear_luminance: bool,
            as_gray: bool,
            input_other: Optional[ImageListIO] = None,
            conversion_type: Literal['xyY_to_sRGB', 'lab_to_sRGB'] = 'xyY_to_sRGB',
            gamut_strategy: str = 'clip',
            verbose: bool = False) -> ImageListIO:
        """Apply the backward color-treatment step to an image collection.

        The function processes the provided input images and optionally incorporates
        additional data (input_other) depending on the context, ensuring that images are
        returned to their original color or grayscale representations.

        Parameters
        ----------
        rec_standard : REC_STANDARD
            Color standard used for processing.

        input_images : ImageListIO
            Main processed buffer to convert back.

        output_images : ImageListIO
            Buffer receiving reconstructed output images.

        linear_luminance : bool
            If True, no perceptual color treatment needs to be undone.

        as_gray : bool
            If True, reconstructs grayscale output.

        input_other : Optional[ImageListIO]
            Auxiliary chromatic data required for color reconstruction.

        conversion_type : Literal["xyY_to_sRGB", "lab_to_sRGB"]
            Backward color conversion to apply.

        gamut_strategy : str
            Strategy for repairing out-of-gamut pixels during conversion.

        verbose : bool
            If True, prints processing messages.

        Raises
        ------
        ValueError
            If color reconstruction requires ``input_other`` but it is missing.

        Returns
        -------
        ImageListIO
            Output images converted back to display-ready representation.
        """
        safe_mode = True
        converter = ColorConverter(rec_standard=rec_standard, safe_mode=safe_mode)

        from shinier.color.GamutControl import GamutControl  # local import avoids circularity
        gamut_control = GamutControl(
            strategy=gamut_strategy,
            rec_standard=rec_standard,
            verbose=verbose,
            color_space='xyY',
        )

        # --- CASE 1: No color treatment ----------------------------------------
        if linear_luminance:
            # Nothing to undo; images already linear or grayscale
            return input_images

        # --- CASE 2: Color treatment branch -----------------------------------
        if not linear_luminance and not as_gray and (input_other is None or input_other.n_channels != 2):
            raise ValueError("input_other should be (H, W, 2) matrices when linear_luminance and as_gray are False")

        for idx, Y in enumerate(input_images):

            # Safety: make sure Y is 2D float64
            Y = Y[..., 0]/255 if Y.ndim == 3 else Y/255

            if not as_gray:
                # Retrieve stored xy if available
                other = input_other[idx]

                # Rebuild xyY or Lab: shape (H, W, 3)
                if conversion_type == "xyY_to_sRGB":
                    # --- Out-of-gamut repair ---
                    Y255, other = gamut_control.apply_image(Y=Y*255, other=other, idx=idx, verbose=verbose)
                    Y = Y255/255.0

                    # Gamma Encode to sRGB
                    output_images[idx] = converter.xyY_to_sRGB(np.dstack((other, Y))) * 255

                elif conversion_type == "lab_to_sRGB":
                    # Convert xyY → sRGB (includes linear→gamma)
                    lab = np.dstack([Y*100, other])
                    output_images[idx] = converter.lab_to_sRGB(lab) * 255

            # Output = Grayscale image
            else:
                # Apply gamma encoding (sRGB transfer function)
                Yg = converter.linRGB_to_sRGB(im3D(Y))

                # Replicate into 3 channels for display compatibility
                output_images[idx] = np.dstack([Yg, Yg, Yg]) * 255

        return output_images


def rgb2gray(image: Union[np.ndarray, Image.Image], conversion_type: RGB_STANDARD = 'equal', matlab_601: bool = False) -> np.ndarray:
    """Convert an R'G'B' image to grayscale luma.

    Parameters
    ----------
    image : Union[np.ndarray, Image.Image]
        RGB image array with a final channel dimension of 3. The image is
        assumed to be gamma-encoded, as with typical sRGB files.

    conversion_type : RGB_STANDARD
        Luma standard to use: ``"equal"``, ``"rec601"``, ``"rec709"``, or
        ``"rec2020"``.

    matlab_601 : bool
        If True and ``conversion_type="rec601"``, uses MATLAB's BT.601 weights.

    Returns
    -------
    np.ndarray
        Grayscale image with the final color channel removed.

    Notes
    -----
    This computes luma (Y') from gamma-encoded components. For physical linear
    luminance, first linearize RGB, combine linear-light coefficients, then
    re-encode if needed.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError(f"Invalid image type {type(image)}. Supported values are Image.Image and np.ndarray")

    if conversion_type not in RGB2GRAY_WEIGHTS.keys():
        raise ValueError('Conversion type must be either rec709, rec601, rec2020 or equal')

    if image.ndim > 2:
        if conversion_type == "equal":
            return np.mean(image[..., :3], axis=-1)
        else:
            weights = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103]) if conversion_type == "rec601" and matlab_601 else RGB2GRAY_WEIGHTS[conversion_type]
            weights /= np.sum(weights)
            return image[..., 0] * weights[0] + image[..., 1] * weights[1] + image[..., 2] * weights[2]
    elif image.ndim == 2:
        return image
    else:
        raise ValueError(f"Invalid image dimension {image.shape}. Supported values are >= 2")


def gray2rgb(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Convert a grayscale image to RGB.

    Parameters
    ----------
    image : Union[np.ndarray, Image.Image]
        Input grayscale image.

    Returns
    -------
    np.ndarray
        RGB image with three identical channels.
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32)
    elif not isinstance(image, np.ndarray):
        raise ValueError(f"Invalid image type {type(image)}. Supported values are Image.Image and np.ndarray")

    if image.ndim > 2:
        image = image[:, :, 0]

    return np.stack((image,) * 3, axis=-1)
