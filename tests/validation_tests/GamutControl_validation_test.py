import numpy as np
import pytest
from shinier.color.GamutControl import GamutControl
from shinier.color import ColorConverter

pytestmark = pytest.mark.validation_tests


# -----------------------------------------------------------------------------
# GamutControl validation tests
# -----------------------------------------------------------------------------

def _assert_in_gamut_linrgb(
    rgb: np.ndarray,
    tol: float = 1e-7,
    allowed_oog_fraction: float = 0.0,
) -> None:
    """Validate that linRGB is (mostly) within [0,1].

    Notes:
        GamutControl optionally allows a small fraction of extreme pixels to remain
        out-of-gamut when `prc_clipping` > 0 (e.g., 0.5%). For validation tests,
        we therefore validate the *fraction* of out-of-gamut pixels rather than
        the global max/min, which is hypersensitive to a handful of outliers.

    Args:
        rgb: Linear RGB array.
        tol: Numerical tolerance for bounds.
        allowed_oog_fraction: Maximum allowed fraction (0..1) of pixels that may
            be out of gamut (any channel <0 or >1 beyond tol).
    """

    rgb = np.asarray(rgb, dtype=np.float64)
    assert np.all(np.isfinite(rgb))
    if rgb.size == 0:
        return

    # Pixel-level out-of-gamut mask (any channel out of bounds).
    oog = (rgb < (0.0 - tol)) | (rgb > (1.0 + tol))
    oog_px = np.any(oog, axis=-1)
    frac_oog = float(np.mean(oog_px))

    assert frac_oog <= allowed_oog_fraction + 1e-6, (
        f"Out-of-gamut fraction too large: {frac_oog:.6f} > {allowed_oog_fraction:.6f}"
    )

    # Robust bound check on the inlier majority.
    # We expect the (1-allowed) quantile to be within bounds.
    q_hi = float(np.quantile(rgb, 1.0 - max(allowed_oog_fraction, 0.0)))
    q_lo = float(np.quantile(rgb, max(allowed_oog_fraction, 0.0)))
    assert q_hi <= 1.0 + tol, f"High quantile out of gamut: {q_hi}"
    assert q_lo >= 0.0 - tol, f"Low quantile out of gamut: {q_lo}"


def _make_synthetic_xyY_from_linrgb(conv: ColorConverter, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert linRGB in [0,1] to (Y_255, xy) suitable for GamutControl."""
    xyz = conv.linRGB_to_xyz(rgb)
    xyY = conv.xyz_to_xyY(xyz)
    x = xyY[..., 0]
    y = xyY[..., 1]
    Y01 = xyY[..., 2]
    Y255 = Y01 * 255.0
    other = np.dstack([x, y])
    return Y255, other


def _reconstruct_linrgb(conv_raw: ColorConverter, Y255: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Reconstruct linRGB from (Y_255, xy) using the raw (no clipping) converter."""
    Y01 = np.asarray(Y255, dtype=np.float64) / 255.0
    x = other[..., 0]
    y = other[..., 1]
    xyY = np.dstack([x, y, Y01])
    xyz = conv_raw.xyY_to_xyz(xyY)
    rgb = conv_raw.xyz_to_linRGB(xyz)
    return rgb


@pytest.mark.validation_tests
@pytest.mark.parametrize("rec_std", ["rec601", "rec709", "rec2020"])
def test_gamutcontrol_constrain_image_luminance_produces_in_gamut(rec_std: str) -> None:
    """Per-image luminance constraint should remove RGB>1 overflows for mid-tones."""
    conv_safe = ColorConverter(rec_standard=rec_std, safe_mode=True)
    conv_raw = ColorConverter(rec_standard=rec_std, safe_mode=False)

    rng = np.random.default_rng(0)
    H, W = 48, 48
    rgb = rng.random((H, W, 3), dtype=np.float64)

    Y255, other = _make_synthetic_xyY_from_linrgb(conv_safe, rgb)

    # Force overflow: boost luminance but keep chromaticity unchanged.
    Y255_over = np.clip(Y255 * 1.8, 0.0, 255.0)

    gamut = GamutControl(
        color_space="xyY",
        strategy="constrain_image_luminance",
        rec_standard=rec_std,
        warning_threshold=1.0,
        prc_clipping=0.5,
        low_Y_desaturate=False,
        low_Y_threshold=0.01,
        low_Y_fade_width=0.0,
        log_low_Y_chroma_loss=False,
    )

    Y255_fixed, other_fixed = gamut.apply_image(Y255_over, other, verbose=False)
    rgb_fixed = _reconstruct_linrgb(conv_raw, Y255_fixed, other_fixed)

    # Validate gamut for mid-tones (ignore near-black where xy is ill-defined).
    Y01_fixed = Y255_fixed / 255.0
    mask = Y01_fixed > 0.05
    # Note: `prc_clipping=0.5` is applied to the *ratio selection* inside GamutControl (a luminance-based
    # criterion). The induced fraction of reconstructed linRGB pixels that fall slightly outside [0,1]
    # can exceed 0.5% because (a) we count a pixel OOG if *any* channel violates the bound, and
    # (b) the mid-tone mask changes the denominator. We therefore allow a small safety margin.
    _assert_in_gamut_linrgb(rgb_fixed[mask], tol=5e-6, allowed_oog_fraction=0.008)


@pytest.mark.validation_tests
@pytest.mark.parametrize("rec_std", ["rec601", "rec709", "rec2020"])
def test_gamutcontrol_constrain_image_chrominance_produces_in_gamut(rec_std: str) -> None:
    """Per-image chrominance constraint should remove RGB over/underflows by desaturation."""
    conv_safe = ColorConverter(rec_standard=rec_std, safe_mode=True)
    conv_raw = ColorConverter(rec_standard=rec_std, safe_mode=False)

    rng = np.random.default_rng(1)
    H, W = 48, 48
    rgb = rng.random((H, W, 3), dtype=np.float64)

    Y255, other = _make_synthetic_xyY_from_linrgb(conv_safe, rgb)

    # Force chroma overflow: scale xy away from the whitepoint.
    W_xyz = conv_safe.white_point
    W_sum = float(np.sum(W_xyz))
    xw = float(W_xyz[0] / W_sum)
    yw = float(W_xyz[1] / W_sum)

    x = other[..., 0]
    y = other[..., 1]
    k = 2.2  # push away from neutral
    other_over = np.dstack([xw + k * (x - xw), yw + k * (y - yw)])

    gamut = GamutControl(
        color_space="xyY",
        strategy="constrain_image_chrominance",
        rec_standard=rec_std,
        warning_threshold=1.0,
        prc_clipping=0.5,
        low_Y_desaturate=False,
        low_Y_threshold=0.01,
        low_Y_fade_width=0.0,
        log_low_Y_chroma_loss=False,
    )

    Y255_fixed, other_fixed = gamut.apply_image(Y255, other_over, verbose=False)
    rgb_fixed = _reconstruct_linrgb(conv_raw, Y255_fixed, other_fixed)

    Y01 = Y255 / 255.0
    mask = Y01 > 0.05
    _assert_in_gamut_linrgb(rgb_fixed[mask], tol=5e-6, allowed_oog_fraction=0.008)


@pytest.mark.validation_tests
@pytest.mark.parametrize("rec_std", ["rec601", "rec709", "rec2020"])
def test_gamutcontrol_dataset_vs_image_luminance_are_consistent(rec_std: str) -> None:
    """Dataset-level luminance constraint should be at least as strict as per-image on the worst case."""
    conv_safe = ColorConverter(rec_standard=rec_std, safe_mode=True)
    conv_raw = ColorConverter(rec_standard=rec_std, safe_mode=False)

    rng = np.random.default_rng(2)
    H, W = 32, 32

    # Make 3 images; force only the last one to be the worst-case overflow.
    rgbs = [rng.random((H, W, 3), dtype=np.float64) for _ in range(3)]
    Ys = []
    others = []
    for rgb in rgbs:
        Y255, other = _make_synthetic_xyY_from_linrgb(conv_safe, rgb)
        Ys.append(Y255)
        others.append(other)

    Ys[2] = np.clip(Ys[2] * 2.0, 0.0, 255.0)

    gamut_ds = GamutControl(
        color_space="xyY",
        strategy="constrain_dataset_luminance",
        rec_standard=rec_std,
        warning_threshold=1.0,
        prc_clipping=0.5,
        low_Y_desaturate=False,
        low_Y_threshold=0.01,
        low_Y_fade_width=0.0,
        log_low_Y_chroma_loss=False,
    )

    # Apply dataset strategy (mutates buffers). Use python lists as a minimal ImageListIO stand-in.
    Ys_ds = [y.copy() for y in Ys]
    others_ds = [o.copy() for o in others]
    Ys_ds, others_ds = gamut_ds.apply_dataset(Ys_ds, others_ds, verbose=False)

    # Apply per-image strategy to each image.
    gamut_im = GamutControl(
        color_space="xyY",
        strategy="constrain_image_luminance",
        rec_standard=rec_std,
        warning_threshold=1.0,
        prc_clipping=0.5,
        low_Y_desaturate=False,
        low_Y_threshold=0.01,
        low_Y_fade_width=0.0,
        log_low_Y_chroma_loss=False,
    )

    Ys_im = []
    for Y255, other in zip(Ys, others):
        Y_fixed, _ = gamut_im.apply_image(Y255, other, verbose=False)
        Ys_im.append(Y_fixed)

    # Dataset scaling factor should be <= the worst per-image factor.
    ratios_im = [float(np.mean(Y_fix / (Y0 + 1e-12))) for Y_fix, Y0 in zip(Ys_im, Ys)]
    ratio_ds_worst = float(np.min([np.mean(Yd / (Y0 + 1e-12)) for Yd, Y0 in zip(Ys_ds, Ys)]))

    assert ratio_ds_worst <= float(np.min(ratios_im)) + 1e-9

    # And dataset-fixed images should reconstruct to in-gamut RGB (mid-tones).
    for Y255_fixed, other_fixed in zip(Ys_ds, others_ds):
        rgb_fixed = _reconstruct_linrgb(conv_raw, Y255_fixed, other_fixed)
        Y01 = Y255_fixed / 255.0
        mask = Y01 > 0.05
        # Dataset-level clipping can yield a slightly larger OOG fraction than the nominal prc_clipping; allow margin.
        _assert_in_gamut_linrgb(rgb_fixed[mask], tol=5e-6, allowed_oog_fraction=0.008)
