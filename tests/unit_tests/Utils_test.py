import numpy as np
import pytest

from shinier.utils import (
    betce_gray,
    classic_he_gray,
    compute_ambe,
    compute_bp2bpsim,
    compute_contrast_improvement,
    compute_image_entropy,
    compute_mssim,
    compute_psnr,
    nfldice_gray,
    rdfhe_gray,
    sfcef_gray,
    tidhe_gray,
)

pytestmark = pytest.mark.unit_tests

IE_METHODS = [
    classic_he_gray,
    tidhe_gray,
    rdfhe_gray,
    nfldice_gray,
    betce_gray,
    sfcef_gray,
]
LUT_METHODS = [classic_he_gray, tidhe_gray, rdfhe_gray, nfldice_gray, betce_gray]


@pytest.mark.parametrize("method", IE_METHODS)
def test_image_enhancement_rejects_non_grayscale(method) -> None:
    """Image-enhancement methods are intentionally limited to 2D grayscale images."""
    with pytest.raises(ValueError, match="2D grayscale"):
        method(np.zeros((4, 4, 1), dtype=np.uint8))


@pytest.mark.parametrize("method", IE_METHODS)
def test_image_enhancement_rejects_non_uint8(method) -> None:
    """Image-enhancement methods use the MATLAB uint8 reference domain."""
    with pytest.raises(ValueError, match="dtype uint8"):
        method(np.zeros((4, 4), dtype=np.float64))


@pytest.mark.parametrize("method", IE_METHODS)
def test_image_enhancement_output_shape_dtype_and_range(method) -> None:
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    result = method(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.min(result) >= 0
    assert np.max(result) <= 255


@pytest.mark.parametrize("method", IE_METHODS)
def test_image_enhancement_constant_image_stable(method) -> None:
    result = method(np.full((64, 64), 128, dtype=np.uint8))
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8
    assert np.min(result) >= 0
    assert np.max(result) <= 255


@pytest.mark.parametrize("method", LUT_METHODS)
def test_image_enhancement_mapping_is_monotonic(method) -> None:
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result = method(image)
    assert np.all(np.diff(result.ravel().astype(np.int64)) >= 0)


def test_classic_he_gray_constant_image_unchanged() -> None:
    image = np.full((16, 16), 128, dtype=np.uint8)
    np.testing.assert_array_equal(classic_he_gray(image), image)


def test_classic_he_gray_two_levels_use_full_range() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    image[:, 8:] = 128
    result = classic_he_gray(image)
    assert np.min(result) == 0
    assert np.max(result) == 255


def test_nfldice_gray_midpoint_maps_to_half_range() -> None:
    """With p_l=127.5, level 128 maps just above 0.5*(L-1), matching MATLAB."""
    image = np.full((8, 8), 128, dtype=np.uint8)
    assert int(nfldice_gray(image)[0, 0]) == 129


def test_contrast_metric_ambe_known_values() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)
    shifted = np.full((4, 4), 10, dtype=np.uint8)
    assert compute_ambe(image, image) == pytest.approx(0.0)
    assert compute_ambe(image, shifted) == pytest.approx(10.0)


def test_contrast_metric_ambe_3d_averages_channels() -> None:
    reference = np.zeros((2, 2, 2), dtype=np.float64)
    enhanced = np.zeros_like(reference)
    enhanced[:, :, 0] = 10
    enhanced[:, :, 1] = -10
    assert compute_ambe(reference, enhanced) == pytest.approx(10.0)


def test_contrast_metric_ci_and_entropy_constant_and_varied() -> None:
    constant = np.zeros((4, 4), dtype=np.uint8)
    varied = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    assert compute_contrast_improvement(constant, n_bins=2) == pytest.approx(0.0)
    assert compute_image_entropy(constant, n_bins=2) == pytest.approx(0.0)
    assert compute_contrast_improvement(varied, n_bins=2) > 0
    assert compute_image_entropy(varied, n_bins=2) == pytest.approx(1.0)


def test_contrast_metric_mssim_identical_is_one() -> None:
    image = np.arange(64, dtype=np.float64).reshape(8, 8)
    assert compute_mssim(image, image, data_range=63) == pytest.approx(1.0)


def test_contrast_metric_psnr_known_values() -> None:
    reference = np.zeros((4, 4), dtype=np.float64)
    enhanced = np.ones((4, 4), dtype=np.float64)

    assert compute_psnr(reference, reference, data_range=1) == np.inf
    assert compute_psnr(reference, enhanced, data_range=1) == pytest.approx(0.0)


def test_contrast_metric_psnr_3d_averages_channels() -> None:
    reference = np.zeros((2, 2, 2), dtype=np.float64)
    enhanced = np.zeros_like(reference)
    enhanced[:, :, 0] = 1
    enhanced[:, :, 1] = 2
    expected = (0.0 + (-10.0 * np.log10(4.0))) / 2.0
    assert compute_psnr(reference, enhanced, data_range=1) == pytest.approx(expected)


def test_contrast_metric_bp2bpsim_known_values() -> None:
    reference = np.array([[0]], dtype=np.uint8)
    enhanced = np.array([[1]], dtype=np.uint8)

    assert compute_bp2bpsim(reference, reference) == pytest.approx(1.0)
    assert compute_bp2bpsim(reference, enhanced) == pytest.approx(7.0 / 8.0)
    assert compute_bp2bpsim(reference, enhanced, n_bits=1) == pytest.approx(0.0)


def test_contrast_metric_bp2bpsim_3d_averages_channels() -> None:
    reference = np.zeros((1, 1, 2), dtype=np.uint8)
    enhanced = np.zeros_like(reference)
    enhanced[:, :, 0] = 1
    assert compute_bp2bpsim(reference, enhanced, n_bits=1) == pytest.approx(0.5)


@pytest.mark.parametrize("metric", [compute_ambe, compute_mssim, compute_psnr, compute_bp2bpsim])
def test_pairwise_contrast_metrics_reject_shape_mismatch(metric) -> None:
    with pytest.raises(ValueError, match="same shape"):
        metric(np.zeros((4, 4)), np.zeros((4, 5)))


def test_contrast_metric_bp2bpsim_rejects_invalid_bit_count() -> None:
    with pytest.raises(ValueError, match="n_bits"):
        compute_bp2bpsim(np.zeros((2, 2)), np.zeros((2, 2)), n_bits=0)
