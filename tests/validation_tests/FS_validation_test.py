"""Property-based validation tests for Floyd–Steinberg error diffusion.

These tests intentionally avoid treating Pillow (or ImageMagick) as bit-exact ground truth.
Instead, they validate theoretical / algorithmic invariants that should hold for a correct
forward-raster Floyd–Steinberg implementation.

Run:
    pytest -q shinier/tests/validation_tests/FS_validation_test.py

Notes:
- The tests are deterministic.
- They should catch common implementation bugs:
    * wrong scaling (0..1 vs 0..255)
    * wrong quantizer / thresholding
    * wrong sign of error
    * wrong tap offsets / wrong padding
    * non-causal propagation (writing into already-quantized pixels)
    * weights not summing to 1 (or not normalized when expected)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
from shinier import utils


def _as_float01(img: np.ndarray) -> np.ndarray:
    """Ensure float64 image in [0,1]."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.integer):
        return utils.uint_to_float01(img, apply_clipping=True).astype(np.float64)
    out = img.astype(np.float64, copy=False)
    if out.min() < 0.0 or out.max() > 1.0:
        raise ValueError("Image must be within [0,1].")
    return out


def _dither_fs(img01: np.ndarray, n_levels: int, *, serpentine: bool = False) -> np.ndarray:
    """Run Floyd–Steinberg via the generic engine, returning levels in [0, n_levels-1]."""
    scan_order = "serpentine" if serpentine else "raster"
    return utils.error_diffusion_dither(
        _as_float01(img01),
        n_levels=n_levels,
        diffusion_map=utils.DiffusionMaps.FLOYD_STEINBERG,
        legacy_mode=False,
        scan_order=scan_order,
        normalize_map=True,
    )


def _reconstruct01(levels: np.ndarray, n_levels: int) -> np.ndarray:
    """Map quantized levels back to float [0,1]."""
    levels_f = np.asarray(levels, dtype=np.float64)
    return levels_f / float(n_levels - 1)


def _make_constant(shape: Tuple[int, int], value: float) -> np.ndarray:
    return np.full(shape, float(value), dtype=np.float64)


def _make_gradient_x(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    x = np.linspace(0.0, 1.0, w, dtype=np.float64)
    return np.tile(x[None, :], (h, 1))


def _crop_center(img: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return img
    return img[margin:-margin, margin:-margin]


def _raster_prefix_mask(h: int, w: int, y0: int, x0: int) -> np.ndarray:
    """Mask of pixels that occur strictly before (y0,x0) in forward raster scan."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return (yy < y0) | ((yy == y0) & (xx < x0))


def test_output_range_and_dtype_binary() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((128, 160), dtype=np.float64)

    out = _dither_fs(img, n_levels=2)

    assert out.dtype.kind in ("u", "i")
    assert out.min() >= 0
    assert out.max() <= 1
    # Stronger condition for binary
    u = np.unique(out)
    assert set(u.tolist()).issubset({0, 1})


def test_output_range_multilevel() -> None:
    rng = np.random.default_rng(1)
    img = rng.random((64, 65), dtype=np.float64)

    n_levels = 8
    out = _dither_fs(img, n_levels=n_levels)

    assert out.dtype.kind in ("u", "i")
    assert out.min() >= 0
    assert out.max() <= (n_levels - 1)


@pytest.mark.parametrize("c1,c2", [(0.1, 0.2), (0.25, 0.75), (0.49, 0.51)])
def test_constant_image_monotonicity_binary(c1: float, c2: float) -> None:
    shape = (256, 256)
    img1 = _make_constant(shape, c1)
    img2 = _make_constant(shape, c2)

    out1 = _dither_fs(img1, n_levels=2)
    out2 = _dither_fs(img2, n_levels=2)

    # Densities must be monotone
    assert float(out1.mean()) <= float(out2.mean())


@pytest.mark.parametrize("c", [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95])
def test_constant_image_density_tracks_input_binary(c: float) -> None:
    # For large constant fields, mean(output) should approximate the input value.
    # This is not a strict theorem for finite images, but deviations should be small.
    shape = (512, 512)
    img = _make_constant(shape, c)

    out = _dither_fs(img, n_levels=2)
    p = float(out.mean())

    # Tolerance: relaxed near extremes; tighter near mid-range
    tol = 0.03 if 0.1 <= c <= 0.9 else 0.05
    assert abs(p - c) <= tol


def test_global_mean_conservation_binary_cropped() -> None:
    # With normalized FS weights, the global mean error should be near 0.
    # However, diffusion pushes error to the future; at the bottom/right borders
    # some error can be lost. Cropping a small margin makes this much tighter.
    rng = np.random.default_rng(2)
    img = rng.random((256, 256), dtype=np.float64)

    out = _dither_fs(img, n_levels=2)
    rec = _reconstruct01(out, n_levels=2)

    margin = 4
    e_full = float((rec - img).mean())
    e_crop = float((_crop_center(rec, margin) - _crop_center(img, margin)).mean())

    # Full-frame mean bias should be small
    assert abs(e_full) <= 0.01
    # Cropped should be tighter
    assert abs(e_crop) <= 0.003


def test_global_mean_conservation_multilevel_cropped() -> None:
    rng = np.random.default_rng(3)
    img = rng.random((256, 256), dtype=np.float64)

    n_levels = 16
    out = _dither_fs(img, n_levels=n_levels)
    rec = _reconstruct01(out, n_levels=n_levels)

    margin = 4
    e_full = float((rec - img).mean())
    e_crop = float((_crop_center(rec, margin) - _crop_center(img, margin)).mean())

    assert abs(e_full) <= 0.01
    assert abs(e_crop) <= 0.003


def test_raster_causality_prefix_invariance() -> None:
    # Causality test: modifying a pixel at (y0,x0) must not change any output pixel
    # that occurs strictly *before* it in forward raster scan.
    shape = (128, 129)
    base = _make_gradient_x(shape)

    y0, x0 = 60, 70
    delta = 0.15

    img_a = base.copy()
    img_b = base.copy()
    img_b[y0, x0] = np.clip(img_b[y0, x0] + delta, 0.0, 1.0)

    out_a = _dither_fs(img_a, n_levels=2)
    out_b = _dither_fs(img_b, n_levels=2)

    mask = _raster_prefix_mask(shape[0], shape[1], y0, x0)
    assert np.array_equal(out_a[mask], out_b[mask])


def test_gradient_column_means_track_expected_binary() -> None:
    # On a smooth gradient, the spatial average per column should track the
    # target ramp (approximately), without large non-monotone artifacts.
    shape = (256, 512)
    img = _make_gradient_x(shape)

    out = _dither_fs(img, n_levels=2)
    col_means = out.mean(axis=0)

    expected = np.linspace(0.0, 1.0, shape[1], dtype=np.float64)

    # Local (per-column) monotonicity is *not* guaranteed for error-diffusion.
    # Floyd–Steinberg can create periodic structures on ramps that cause small
    # local dips in column means (especially at finite height). What we do expect
    # is that the *low-frequency trend* is monotone.
    #
    # So we test monotonicity on a moving-average smoothed profile.
    win = 33  # odd window; smooth over FS periodicity on ramps
    kernel = np.ones(win, dtype=np.float64) / float(win)

    # Avoid edge artifacts from convolution ("same" implicitly pads with zeros).
    # Use edge padding and a valid convolution to keep length and preserve the ramp ends.
    pad = win // 2
    padded = np.pad(col_means, (pad, pad), mode="edge")
    col_means_smooth = np.convolve(padded, kernel, mode="valid")

    # Even after smoothing, very small local dips can remain; we only enforce
    # monotonicity on the interior where border error-loss and padding effects
    # are minimal.
    margin = 2 * pad
    diffs = np.diff(col_means_smooth[margin:-margin])
    assert float(diffs.min()) >= -0.02

    # Correlation should be very high
    corr = np.corrcoef(col_means, expected)[0, 1]
    assert float(corr) >= 0.995

    # RMSE should be small
    rmse = float(np.sqrt(np.mean((col_means - expected) ** 2)))
    assert rmse <= 0.03


def test_serpentine_changes_pattern_but_preserves_density() -> None:
    # Serpentine scan should change the pattern but preserve the overall density.
    rng = np.random.default_rng(4)
    img = rng.random((256, 256), dtype=np.float64)

    out_raster = _dither_fs(img, n_levels=2, serpentine=False)
    out_serp = _dither_fs(img, n_levels=2, serpentine=True)

    # Not necessarily always different, but very likely.
    assert not np.array_equal(out_raster, out_serp)

    # Density should be very close
    assert abs(float(out_raster.mean()) - float(out_serp.mean())) <= 0.01


def test_weights_sum_to_one() -> None:
    # Sanity check on the configured diffusion map.
    dm = np.asarray(utils.DiffusionMaps.FLOYD_STEINBERG, dtype=np.float64)
    wsum = float(dm[:, 2].sum())
    assert abs(wsum - 1.0) <= 1e-12
