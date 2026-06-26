"""
Unit tests for shinier.color.converter.ColorConverter

These tests validate the internal consistency and Pydantic integration of the
ColorConverter class, without depending on the external `colour-science` package.

Each test ensures that configuration, validation, and core reversible pipelines
(sRGB↔linRGB↔XYZ↔Lab↔xyY) behave as expected numerically.

Run via:
    pytest -v tests/unit_tests/test_color_converter_unit.py
"""

import numpy as np
import pytest
from shinier.ImageListIO import ImageListIO
from shinier.color import ColorConverter, ColorTreatment, WHITE_D65, COLOR_STANDARDS, rgb2gray, rgb2ntsc_intensity

pytestmark = pytest.mark.unit_tests


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(params=["rec601", "rec709", "rec2020"])
def converter(request) -> ColorConverter:
    """Fixture returning a configured ColorConverter for each Rec. standard."""
    return ColorConverter(rec_standard=request.param)


@pytest.fixture
def rgb_sample() -> np.ndarray:
    """Fixture returning a small, reproducible sRGB array in [0, 1]."""
    rng = np.random.default_rng(1234)
    return rng.random((8, 8, 3), dtype=np.float64)


# -----------------------------------------------------------------------------
# Model construction and configuration
# -----------------------------------------------------------------------------
def test_color_converter_pydantic_integration(converter: ColorConverter):
    """Ensure Pydantic correctly initializes dependent attributes."""
    cfg = COLOR_STANDARDS[converter.rec_standard]

    # Gamma and matrices must match the reference configuration
    assert np.isclose(converter.gamma, cfg["gamma"])
    assert np.allclose(converter.white_point, WHITE_D65)
    assert np.allclose(converter.M_RGB2XYZ, cfg["M_RGB2XYZ"])
    assert np.allclose(
        np.linalg.inv(converter.M_RGB2XYZ), converter.M_XYZ2RGB, atol=1e-12
    )


def test_invalid_standard_raises():
    """Ensure unsupported color standards raise validation errors."""
    with pytest.raises(ValueError):
        ColorConverter(rec_standard="invalid_std")


def test_white_point_is_independent():
    """Ensure white_point is a copy, not a shared mutable array."""
    c1 = ColorConverter(rec_standard="rec709")
    c2 = ColorConverter(rec_standard="rec709")
    c1.white_point[0] += 0.1
    assert not np.allclose(c1.white_point, c2.white_point)


# -----------------------------------------------------------------------------
# Basic transformation consistency
# -----------------------------------------------------------------------------
def test_srgb_linrgb_roundtrip(converter: ColorConverter, rgb_sample: np.ndarray):
    """Validate that sRGB↔linRGB round-trip maintains numerical consistency."""
    lin = converter.sRGB_to_linRGB(rgb_sample)
    rgb_rec = converter.linRGB_to_sRGB(lin)
    assert np.all(np.isfinite(lin))
    assert np.allclose(rgb_sample, rgb_rec, atol=1e-10)


def test_linrgb_xyz_roundtrip(converter: ColorConverter, rgb_sample: np.ndarray):
    """Ensure linRGB↔XYZ transforms are numerically invertible."""
    lin = converter.sRGB_to_linRGB(rgb_sample)
    xyz = converter.linRGB_to_xyz(lin)
    lin_rec = converter.xyz_to_linRGB(xyz)
    assert xyz.shape == rgb_sample.shape
    assert np.allclose(lin, lin_rec, atol=1e-10)


def test_xyz_lab_roundtrip(converter: ColorConverter, rgb_sample: np.ndarray):
    """Ensure XYZ↔Lab transforms are reversible."""
    xyz = converter.sRGB_to_xyz(rgb_sample)
    lab = converter.xyz_to_lab(xyz)
    xyz_rec = converter.lab_to_xyz(lab)
    assert lab.shape == xyz.shape
    assert np.allclose(xyz, xyz_rec, atol=1e-8)


def test_xyY_xyz_roundtrip(converter: ColorConverter, rgb_sample: np.ndarray):
    """Ensure xyY↔XYZ conversion is reversible."""
    xyz = converter.sRGB_to_xyz(rgb_sample)
    xyY = converter.xyz_to_xyY(xyz)
    xyz_rec = converter.xyY_to_xyz(xyY)
    assert xyY.shape == xyz.shape
    assert np.allclose(xyz, xyz_rec, atol=1e-8)


def test_full_pipeline_roundtrip(converter: ColorConverter, rgb_sample: np.ndarray):
    """Validate the full sRGB↔Lab pipeline is stable within numerical precision."""
    lab = converter.sRGB_to_lab(rgb_sample)
    rgb_rec = converter.lab_to_sRGB(lab)
    assert rgb_rec.shape == rgb_sample.shape
    assert np.allclose(rgb_sample, rgb_rec, atol=1e-4)


# -----------------------------------------------------------------------------
# Edge case and stability tests
# -----------------------------------------------------------------------------
def test_xyz_to_xyY_division_safety(converter: ColorConverter):
    """Ensure xyz_to_xyY handles division-by-zero safely."""
    xyz = np.zeros((2, 2, 3))
    xyY = converter.xyz_to_xyY(xyz)
    assert np.isfinite(xyY).all()
    assert np.all(xyY[..., 2] == 0)  # Y should be preserved


def test_xyY_to_xyz_division_safety(converter: ColorConverter):
    """Ensure xyY_to_xyz handles zero denominators correctly."""
    xyY = np.zeros((2, 2, 3))
    xyz = converter.xyY_to_xyz(xyY)
    assert np.isfinite(xyz).all()


def test_lab_monotonicity(converter: ColorConverter, rgb_sample: np.ndarray):
    """Basic sanity: higher luminance in XYZ should yield higher L* in Lab."""
    xyz = converter.sRGB_to_xyz(rgb_sample)
    lab = converter.xyz_to_lab(xyz)
    assert np.corrcoef(xyz[..., 1].ravel(), lab[..., 0].ravel())[0, 1] > 0.95


def test_repr_and_assignment_behavior():
    """Ensure assignment triggers Pydantic re-validation."""
    c = ColorConverter(rec_standard="rec709")
    c.rec_standard = "rec601"
    assert np.isclose(c.gamma, COLOR_STANDARDS["rec601"]["gamma"])
    assert np.allclose(c.M_RGB2XYZ, COLOR_STANDARDS["rec601"]["M_RGB2XYZ"])


def test_rgb2ntsc_intensity_uses_matlab_yiq_weights(rgb_sample: np.ndarray):
    """NTSC intensity is the Y channel from MATLAB's rgb2ntsc/YIQ transform."""
    weights = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])
    direct = rgb2gray(rgb_sample, weighting_standard="rec601", matlab_601=True)

    np.testing.assert_allclose(direct, np.tensordot(rgb_sample, weights, axes=([-1], [0])))
    np.testing.assert_allclose(rgb2ntsc_intensity(rgb_sample), direct)


def test_legacy_grayscale_forward_uses_matlab_ntsc_intensity(rgb_sample: np.ndarray):
    """Legacy grayscale color treatment uses MATLAB-compatible NTSC/YIQ intensity."""
    image = (rgb_sample * 255).astype(np.float64)
    images = ImageListIO(input_data=[image], conserve_memory=False)

    result, other = ColorTreatment.forward_color_treatment(
        rec_standard="rec601",
        input_images=images,
        output_images=images,
        linear_luminance=False,
        as_gray=True,
        legacy_mode=True,
    )

    np.testing.assert_allclose(result[0], rgb2ntsc_intensity(image))
    assert other is None


def test_legacy_grayscale_backward_does_not_gamma_encode() -> None:
    """Legacy grayscale output keeps MATLAB-compatible intensities unchanged."""
    gray = np.array([[0.0, 64.0], [128.0, 255.0]], dtype=np.float64)
    images = ImageListIO(input_data=[gray], conserve_memory=False)

    result = ColorTreatment.backward_color_treatment(
        rec_standard="rec601",
        input_images=images,
        output_images=images,
        linear_luminance=False,
        as_gray=True,
        legacy_mode=True,
    )

    np.testing.assert_allclose(result[0], np.dstack([gray, gray, gray]))


def test_legacy_grayscale_backward_skips_gamma_non_legacy_applies_it() -> None:
    """Explicitly verify that legacy_mode controls gamma encoding in the grayscale backward pass.

    legacy_mode=True must return Y values unchanged (no sRGB transfer function).
    legacy_mode=False must apply linRGB_to_sRGB, producing different values for mid-range inputs.
    If the legacy_mode branch is missing or broken both results would be identical.
    """
    from shinier.color.Converter import ColorConverter
    # Mid-range linear values where gamma makes a measurable difference (not 0 or 1)
    gray = np.array([[50.0, 100.0], [150.0, 200.0]], dtype=np.float64)

    images_legacy = ImageListIO(input_data=[gray.copy()], conserve_memory=False)
    images_non_legacy = ImageListIO(input_data=[gray.copy()], conserve_memory=False)

    result_legacy = ColorTreatment.backward_color_treatment(
        rec_standard="rec601", input_images=images_legacy, output_images=images_legacy,
        linear_luminance=False, as_gray=True, legacy_mode=True,
    )
    result_non_legacy = ColorTreatment.backward_color_treatment(
        rec_standard="rec601", input_images=images_non_legacy, output_images=images_non_legacy,
        linear_luminance=False, as_gray=True, legacy_mode=False,
    )

    # legacy_mode=True: output must equal input replicated (no gamma applied)
    np.testing.assert_allclose(result_legacy[0], np.dstack([gray, gray, gray]), atol=1e-6,
                               err_msg="legacy_mode=True must leave Y values unchanged")

    # legacy_mode=False: output must equal gamma-encoded Y replicated
    converter = ColorConverter(rec_standard="rec601")
    Y_linear = gray / 255.0
    Y_gamma = converter.linRGB_to_sRGB(Y_linear[..., np.newaxis])[..., 0] * 255
    np.testing.assert_allclose(result_non_legacy[0], np.dstack([Y_gamma, Y_gamma, Y_gamma]), atol=1e-6,
                               err_msg="legacy_mode=False must apply sRGB gamma encoding")

    # The two must differ — if the branch was missing this would fail
    assert not np.allclose(result_legacy[0], result_non_legacy[0]), \
        "legacy_mode=True and legacy_mode=False must produce different outputs for mid-range values"
