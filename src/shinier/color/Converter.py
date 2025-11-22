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
from typing import Literal, Union, Optional, TYPE_CHECKING, Tuple
from pydantic import Field, ConfigDict, model_validator
from PIL import Image
from shinier.utils import im3D
from shinier.base import InformativeBaseModel
from shinier import REPO_ROOT

if TYPE_CHECKING:
    from shinier.ImageListIO import ImageListIO

# D65 white point normalized to Y=1.0
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

    Attributes:
        standard (str): Rec. standard ("rec601", "rec709", or "rec2020").
        gamma (float): Transfer-function exponent (≈2.2–2.4).
        white_point (np.ndarray): Reference white (default: D65).
        M_RGB2XYZ (np.ndarray): Forward RGB→XYZ matrix.
        M_XYZ2RGB (np.ndarray): Inverse XYZ→RGB matrix.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    standard: Literal["rec601", "rec709", "rec2020"] = "rec709"
    gamma: float = 2.4
    white_point: np.ndarray = Field(default_factory=lambda: WHITE_D65.copy())
    M_RGB2XYZ: np.ndarray = Field(default_factory=lambda: M_RGB2XYZ_709.copy())
    M_XYZ2RGB: np.ndarray = Field(default_factory=lambda: np.linalg.inv(M_RGB2XYZ_709))

    # --- Pydantic-setting of attributes as a function of standard ---
    @model_validator(mode="after")
    def apply_standard_config(self) -> "ColorConverter":
        cfg = COLOR_STANDARDS[self.standard]
        object.__setattr__(self, "gamma", cfg["gamma"])
        object.__setattr__(self, "white_point", cfg["white"].copy())
        object.__setattr__(self, "M_RGB2XYZ", cfg["M_RGB2XYZ"].copy())
        object.__setattr__(self, "M_XYZ2RGB", np.linalg.inv(cfg["M_RGB2XYZ"]))
        return self

    # ------------------------------------------------------------------
    # sRGB ↔ linRGB
    # ------------------------------------------------------------------
    def sRGB_to_linRGB(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.clip(rgb, 0, 1)
        if self.standard == "rec2020":
            alpha, beta = 1.0993, 0.0181
            return np.where(rgb < beta * 4.5, rgb / 4.5, ((rgb + (alpha - 1)) / alpha) ** (1 / 0.45))
        else:
            return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** self.gamma)

    def linRGB_to_sRGB(self, linRGB: np.ndarray) -> np.ndarray:
        linRGB = np.clip(linRGB, 0, 1)
        if self.standard == "rec2020":
            alpha, beta = 1.0993, 0.0181
            return np.where(linRGB < beta, 4.5 * linRGB, alpha * np.power(linRGB, 0.45) - (alpha - 1))
        else:
            return np.where(linRGB <= 0.0031308, 12.92 * linRGB, 1.055 * np.power(linRGB, 1 / self.gamma) - 0.055)

    # ------------------------------------------------------------------
    # linRGB ↔ XYZ
    # ------------------------------------------------------------------
    def linRGB_to_xyz(self, linRGB: np.ndarray) -> np.ndarray:
        out = np.empty_like(linRGB)
        np.matmul(linRGB.reshape(-1, 3), self.M_RGB2XYZ.T, out=out.reshape(-1, 3))  # linRGB @ self.M_RGB2XYZ.T
        return out

    def xyz_to_linRGB(self, xyz: np.ndarray) -> np.ndarray:
        out = np.empty_like(xyz)
        np.matmul(xyz.reshape(-1, 3), self.M_XYZ2RGB.T, out=out.reshape(-1, 3))  # xyz @ self.M_XYZ2RGB.T
        return np.clip(out, 0, 1, out=out)

    # ------------------------------------------------------------------
    # XYZ ↔ xyY
    # ------------------------------------------------------------------
    @staticmethod
    def xyz_to_xyY(xyz: np.ndarray) -> np.ndarray:
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        denom = X + Y + Z
        denom_safe = np.where(denom == 0, 1.0, denom)
        x, y = X / denom_safe, Y / denom_safe
        return np.stack([x, y, Y], axis=-1)

    @staticmethod
    def xyY_to_xyz(xyY: np.ndarray) -> np.ndarray:
        x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
        y_safe = np.where(y == 0, 1.0, y)
        X = x * Y / y_safe
        Z = (1 - x - y) * Y / y_safe
        return np.stack([X, Y, Z], axis=-1)

    # ------------------------------------------------------------------
    # XYZ ↔ Lab
    # ------------------------------------------------------------------
    @staticmethod
    def _f_lab(t: np.ndarray) -> np.ndarray:
        epsilon, kappa = 216 / 24389, 24389 / 27
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16) / 116)

    @staticmethod
    def _f_lab_inv(t: np.ndarray) -> np.ndarray:
        epsilon, kappa = 216 / 24389, 24389 / 27
        return np.where(t > (kappa * epsilon + 16) / 116, t**3, (116 * t - 16) / kappa)

    def xyz_to_lab(self, xyz: np.ndarray) -> np.ndarray:
        xyz_n = xyz / self.white_point
        f = self._f_lab(xyz_n)
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1)

    def lab_to_xyz(self, lab: np.ndarray) -> np.ndarray:
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
        return self.linRGB_to_xyz(self.sRGB_to_linRGB(rgb))

    def xyz_to_sRGB(self, xyz: np.ndarray) -> np.ndarray:
        return self.linRGB_to_sRGB(self.xyz_to_linRGB(xyz))

    def sRGB_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        return self.xyz_to_lab(self.sRGB_to_xyz(rgb))

    def lab_to_sRGB(self, lab: np.ndarray) -> np.ndarray:
        return self.xyz_to_sRGB(self.lab_to_xyz(lab))

    def sRGB_to_xyY(self, rgb: np.ndarray) -> np.ndarray:
        return self.xyz_to_xyY(self.sRGB_to_xyz(rgb))

    def xyY_to_sRGB(self, xyY: np.ndarray) -> np.ndarray:
        return self.xyz_to_sRGB(self.xyY_to_xyz(xyY))


class ColorTreatment(ColorConverter):

    @staticmethod
    def forward_color_treatment(
            rec_standard: REC_STANDARD,
            input_images: ImageListIO,
            output_images: ImageListIO,
            linear_luminance: bool,
            as_gray: bool,
            output_other: Optional[ImageListIO] = None,
            conversion_type: Literal['sRGB_to_xyY', 'sRGB_to_lab'] = 'sRGB_to_xyY',
            legacy_mode: bool = False,
) -> Tuple[ImageListIO, Optional[ImageListIO]]:
        """
        Processes a list of images with an optional color treatment conversion or transformation.

        This static method performs a forward pass based on the specified color treatment, grayscale
        conversion, and color space transformation. Depending on the input parameters, the images
        may undergo various treatments such as conversion to grayscale, transformation into a different
        color space (sRGB to xyY or sRGB to Lab), or extracting specific channels for processing.

        Args:
            rec_standard (REC_STANDARD): Reference color standard for the image processing.
            input_images (ImageListType): List of input images to process.
            output_images (ImageListType): Temporary buffer to store the processed image data, must match the
                structure of `images`.
            output_other (Optional[ImageListType]): Secondary buffer to store optional components (if applicable).
            linear_luminance (bool): Defines whether the color treatment is applied.
                If False, color treatment is enabled.
            as_gray (bool): Determines whether output should be a grayscale
                image (True) or a color image (False).
            conversion_type (Literal['sRGB_to_xyY', 'sRGB_to_lab']): Specifies the type of color space
                conversion to apply. Defaults to 'sRGB_to_xyY'.
            legacy_mode (bool): If True, uses matlab rgb2gray converter.

        Returns:
            ImageListType: Processed set of images after the selected transformation or treatment. If
            color treatment is enabled and `as_gray` is 0, two buffers are returned: one containing the
            luminance channel and the other containing auxiliary channels.

        Raises:
            ValueError: If `linear_luminance` is False and `output_other` is not provided.
        """
        if not linear_luminance and not as_gray and output_other is None:
            raise ValueError("output_other cannot be None when linear_luminance is False and as_gray is false")

        converter = ColorConverter(standard=rec_standard)

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
                    _image = converter.sRGB_to_xyY(image / 255)

                    # Extract luminance (Y) channel — for processing
                    output_images[idx] = _image[:, :, 2] * 255

                    # Optionally store x and y for later reconstruction
                    if as_gray == 0:
                        output_other[idx] = _image[:, :, :2]

                elif conversion_type == "sRGB_to_lab":
                    # Convert from sRGB → xyY (internally handles gamma decoding)
                    _image = converter.sRGB_to_lab(image / 255)

                    # Extract luminance (Y) channel — for processing
                    output_images[idx] = _image[:, :, 0]/100 * 255

                    # Optionally store x and y for later reconstruction
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
            conversion_type: Literal['xyY_to_sRGB', 'lab_to_sRGB'] = 'xyY_to_sRGB') -> ImageListIO:
        """
        Reverts color treatments applied previously to a set of images. This operation is
        performed based on the defined color treatment type and conversion method.

        The function processes the provided input images and optionally incorporates
        additional data (input_other) depending on the context, ensuring that images are
        returned to their original color or grayscale representations.

        Static Method:
            backward_color_treatment

        Args:
            rec_standard (REC_STANDARD): The color space standard used for processing
                (e.g., REC709, REC2020).
            input_images (ImageListType): A list or array of images that have undergone prior
                color treatment and need restoration.
            output_images (ImageListType): Temporary buffer to store the processed image data, must match the
                structure of `images`.
            input_other (Optional[ImageListType]): Additional arrays containing auxiliary
                data required for certain reconversions. Mandatory if `linear_luminance=1`.
            linear_luminance (bool): Indicator of whether color treatment was
                applied; 0 means no treatment, 1 means treatment was applied.
            as_gray (bool): Determines whether output should be a grayscale
                image (True) or a color image (False).
            conversion_type (Literal['xyY_to_sRGB', 'lab_to_sRGB']): Specifies the method
                for reconverting colors: either 'xyY_to_sRGB' to transform xyY to sRGB,
                or 'lab_to_sRGB' to transform Lab to sRGB. Defaults to 'xyY_to_sRGB'.

        Raises:
            ValueError: If `linear_luminance` is False and `input_other` is None.

        Returns:
            ImageListType: The processed set of images converted back to their original
            color space or gamma-encoded representation.
        """
        converter = ColorConverter(standard=rec_standard)

        # --- CASE 1: No color treatment ----------------------------------------
        if linear_luminance:
            # Nothing to undo; images already linear or grayscale
            return input_images

        # --- CASE 2: Color treatment branch -----------------------------------
        if not linear_luminance and not as_gray and (input_other is None or input_other.n_channels != 2):
            raise ValueError("input_other should be (H, W, 2) matrices when linear_luminance == 1 and as_gray is False")

        for idx, Y in enumerate(input_images):

            # Safety: make sure Y is 2D float64
            Y = Y[..., 0]/255 if Y.ndim == 3 else Y/255

            if not as_gray:
                # Retrieve stored xy if available
                other = input_other[idx]

                # Rebuild xyY or Lab: shape (H, W, 3)
                if conversion_type == "xyY_to_sRGB":
                    # Convert xyY → sRGB (includes linear→gamma)
                    xyY = np.dstack([other, Y])
                    output_images[idx] = converter.xyY_to_sRGB(xyY) * 255
                if conversion_type == "lab_to_sRGB":
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
    """
    Convert an R'G'B' image to grayscale (luma, Y′) using ITU luma coefficients.

    Args:
        image (np.ndarray or Image.Image):
            RGB image array with last dimension = 3. Assumed to be gamma-encoded R′G′B′ (i.e., not linear light), which
            matches typical sRGB/Rec.709-style images loaded from files (e.g. png or jpg images).
        conversion_type : {"equal", "rec601", "rec709", "rec2020"}, default "rec709"
            Choice of luma standard:
              - "equal" → Y′ = 0.333 R′ + 0.333 G′ + 0.333 B′
              - "rec601" → Y′ = 0.299 R′ + 0.587 G′ + 0.114 B′
              - "rec709" → Y′ = 0.2125 R′ + 0.7154 G′ + 0.0721 B′
              - "rec2020" → Y′ = 0.2627 R′ + 0.6780 G′ + 0.0593 B′
        matlab_601 (bool, optional): If true, uses weights for Matlab's rgb2gray Rec.ITU-R BT.601-7 version.

    Returns:
        gray : np.ndarray
            Grayscale image (same shape as input but last channel removed).

    Notes:
        - This computes luma (Y′) from gamma-encoded components, as defined by the ITU
          matrices for Y′CbCr / Y′CbcCrc. For physical linear luminance, you would need
          to first linearize R′G′B′ using the appropriate transfer function,
          mix with linear-light coefficients, then re-encode if desired.
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
    """
    Convert a grayscale image to RGB.

    Args:
        image (Union[np.ndarray, Image.Image]): The input grayscale image.

    Returns:
        np.ndarray: The RGB image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32)
    elif not isinstance(image, np.ndarray):
        raise ValueError(f"Invalid image type {type(image)}. Supported values are Image.Image and np.ndarray")

    if image.ndim > 2:
        image = image[:, :, 0]

    return np.stack((image,) * 3, axis=-1)
