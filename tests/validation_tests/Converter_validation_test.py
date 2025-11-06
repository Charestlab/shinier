"""
Validation of Shinier ColorConverter against colour-science.

This test suite validates the correctness of Shinier’s color-space
transformations across Rec.601, Rec.709, and Rec.2020 standards.

Dependencies:
    pip install colour-science pytest numpy
"""

import numpy as np
import pytest
import colour
from shinier.color import ColorConverter
from colour.models.rgb.transfer_functions.srgb import eotf_sRGB, eotf_inverse_sRGB
import copy

pytestmark = pytest.mark.validation_tests

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def mse(a, b):
    """Mean squared error."""
    return np.mean((a - b) ** 2)


def max_abs_err(a, b):
    """Maximum absolute deviation."""
    return np.max(np.abs(a - b))


# -----------------------------------------------------------------------------
# Mapping between Shinier and colour-science Rec. standards
# -----------------------------------------------------------------------------
REC_MAP = {
    "rec601": "ITU-R BT.470 - 625",  # Rec.601 equivalent
    "rec709": "ITU-R BT.709",
    "rec2020": "ITU-R BT.2020",
}




# -----------------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------------
# @pytest.mark.parametrize("rec_std", ["rec2020"])
@pytest.mark.parametrize("rec_std", ["rec601", "rec709", "rec2020"])
def test_color_converter_against_colour(rec_std):
    """Validate all color-space transformations against colour-science."""
    conv = ColorConverter(standard=rec_std)
    cs = colour.RGB_COLOURSPACES[REC_MAP[rec_std]]
    cs_customized = copy.deepcopy(cs)
    if rec_std == "rec709":
        cs_customized.cctf_decoding = eotf_sRGB
        cs_customized.cctf_encoding = eotf_inverse_sRGB
    elif rec_std == "rec601":
        cs_customized.cctf_decoding = conv.sRGB_to_linRGB
        cs_customized.cctf_encoding = conv.linRGB_to_sRGB

    # Random sRGB/Rec image (values in [0, 1])
    rng = np.random.default_rng(42)
    rgb = rng.random((32, 32, 3), dtype=np.float64)

    # -------------------------------------------------------------------------
    # sRGB ↔ linRGB → sRGB (round-trip)
    # -------------------------------------------------------------------------
    linRGB = conv.sRGB_to_linRGB(rgb)
    rgb_rec = conv.linRGB_to_sRGB(linRGB)
    assert mse(rgb, rgb_rec) < 1e-10, "sRGB↔linRGB round-trip error too large"

    # -------------------------------------------------------------------------
    # sRGB ↔ linRGB (compare with colour-science)
    # -------------------------------------------------------------------------
    rgb_lin_ref = cs_customized.cctf_decoding(rgb)  # matches your sRGB constants
    assert mse(linRGB, rgb_lin_ref) < 1e-8, f"sRGB↔linRGB mismatch ({rec_std})"
    rgb_rec2 = cs_customized.cctf_encoding(rgb_lin_ref)  # matches your sRGB constants
    assert mse(rgb_rec2, rgb_rec) < 1e-10, f"sRGB↔linRGB mismatch ({rec_std})"

    # -------------------------------------------------------------------------
    # linRGB ↔ XYZ (compare with colour-science + round-trip)
    # -------------------------------------------------------------------------
    xyz_ref = colour.RGB_to_XYZ(
        linRGB,
        cs_customized,
        illuminant=cs_customized.whitepoint,
        apply_cctf_decoding=False  # already linear
    )
    xyz_own = conv.linRGB_to_xyz(linRGB)
    assert mse(xyz_ref, xyz_own) < 1e-10, f"linRGB→XYZ mismatch ({rec_std})"

    linRGB_ref = colour.XYZ_to_RGB(
        xyz_ref,
        cs_customized,
        illuminant=cs_customized.whitepoint,
        apply_cctf_encoding=False  # stay linear
    )
    linRGB_own = conv.xyz_to_linRGB(xyz_own)
    assert mse(linRGB_ref, linRGB_own) < 1e-10, f"XYZ→linRGB mismatch ({rec_std})"
    assert mse(linRGB_ref, linRGB) < 1e-10, f"linRGB round-trip mismatch ({rec_std})"

    linRGB_rec = conv.xyz_to_linRGB(xyz_own)
    assert mse(linRGB, linRGB_rec) < 1e-12, "linRGB↔XYZ round-trip error too large"

    # -------------------------------------------------------------------------
    # XYZ ↔ Lab (compare with colour-science + round-trip)
    # -------------------------------------------------------------------------
    # --- Forward: XYZ → Lab ---
    lab_ref = colour.XYZ_to_Lab(xyz_own, cs_customized.whitepoint)
    lab_own = conv.xyz_to_lab(xyz_own)
    assert mse(lab_ref, lab_own) < 1e-4, f"XYZ→Lab mismatch ({rec_std})"

    # --- Backward: Lab → XYZ ---
    xyz_ref = colour.Lab_to_XYZ(lab_own, cs_customized.whitepoint)
    xyz_own = conv.lab_to_xyz(lab_own)
    assert mse(xyz_ref, xyz_own) < 1e-8, f"Lab→XYZ mismatch ({rec_std})"

    # --- Round-trip consistency check ---
    xyz_round = conv.lab_to_xyz(conv.xyz_to_lab(xyz_own))
    assert mse(xyz_own, xyz_round) < 1e-10, f"XYZ↔Lab round-trip error too large ({rec_std})"

    # -------------------------------------------------------------------------
    # sRGB ↔ XYZ  (compare with colour-science + round-trip)
    # -------------------------------------------------------------------------
    xyz_own = conv.sRGB_to_xyz(rgb)
    xyz_ref = colour.RGB_to_XYZ(
        rgb,
        cs_customized,
        illuminant=cs_customized.whitepoint,
        apply_cctf_decoding=True,
    )
    assert mse(xyz_own, xyz_ref) < 1e-8, f"sRGB→XYZ MSE too large ({rec_std})"

    rgb_own = conv.xyz_to_sRGB(xyz_own)
    rgb_ref = colour.XYZ_to_RGB(
        xyz_ref,
        cs_customized,
        illuminant=cs_customized.whitepoint,
        apply_cctf_encoding=True,
    )
    assert mse(rgb_own, rgb_ref) < 1e-4, f"XYZ→sRGB MSE too large ({rec_std})"

    rgb_round = conv.xyz_to_sRGB(conv.sRGB_to_xyz(rgb))
    assert mse(rgb, rgb_round) < 1e-8, f"sRGB↔XYZ round-trip MSE too large ({rec_std})"

    # -------------------------------------------------------------------------
    # sRGB ↔ Lab  (compare with colour-science + round-trip)
    # -------------------------------------------------------------------------
    lab_own = conv.sRGB_to_lab(rgb)
    xyz_ref = colour.RGB_to_XYZ(
        rgb,
        cs_customized,
        illuminant=cs_customized.whitepoint,
        apply_cctf_decoding=True,
    )
    lab_ref = colour.XYZ_to_Lab(xyz_ref, cs_customized.whitepoint)
    assert mse(lab_own, lab_ref) < 1e-3, f"sRGB→Lab MSE too large ({rec_std})"

    rgb_own = conv.lab_to_sRGB(lab_own)
    xyz_back_ref = colour.Lab_to_XYZ(lab_ref, cs_customized.whitepoint)
    rgb_ref = colour.XYZ_to_RGB(
        xyz_back_ref,
        cs_customized,
        illuminant=cs_customized.whitepoint,
        apply_cctf_encoding=True,
    )
    assert mse(rgb_own, rgb_ref) < 1e-3, f"Lab→sRGB MSE too large ({rec_std})"

    rgb_round = conv.lab_to_sRGB(conv.sRGB_to_lab(rgb))
    assert mse(rgb, rgb_round) < 1e-8, f"sRGB↔Lab round-trip MSE too large ({rec_std})"
