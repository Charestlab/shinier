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
from shinier.color import ColorConverter, WHITE_D65, COLOR_STANDARDS

pytestmark = pytest.mark.unit_tests

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(params=["rec601", "rec709", "rec2020"])
def converter(request) -> ColorConverter:
    """Fixture returning a configured ColorConverter for each Rec. standard."""
    return ColorConverter(standard=request.param)


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
    cfg = COLOR_STANDARDS[converter.standard]

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
        ColorConverter(standard="invalid_std")


def test_white_point_is_independent():
    """Ensure white_point is a copy, not a shared mutable array."""
    c1 = ColorConverter(standard="rec709")
    c2 = ColorConverter(standard="rec709")
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
    c = ColorConverter(standard="rec709")
    c.standard = "rec601"
    assert np.isclose(c.gamma, COLOR_STANDARDS["rec601"]["gamma"])
    assert np.allclose(c.M_RGB2XYZ, COLOR_STANDARDS["rec601"]["M_RGB2XYZ"])