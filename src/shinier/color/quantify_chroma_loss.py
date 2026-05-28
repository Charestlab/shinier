from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union, List, Dict, Any

import numpy as np

from shinier.color import ColorTreatment, REC_STANDARD


# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class ChromaInfoLossResult:
    """Entropy-only chroma information loss result for one ``y1`` value.

    This container is kept for backward compatibility with earlier chroma-loss
    analyses.

    Runtime Attributes
    ------------------
    - ``y1`` (float): Low-luminance threshold or tested Y value.
    - ``h_before_bpp`` (float): Chroma entropy before treatment, in bits per
      pixel.
    - ``h_after_bpp`` (float): Chroma entropy after treatment, in bits per
      pixel.
    - ``loss_bpp`` (float): Entropy loss in bits per pixel.
    - ``frac_pixels_affected`` (float): Fraction of pixels affected, in [0, 1].
    - ``n_pixels_measured`` (int): Number of pixels included in the measurement.
    """

    y1: float
    h_before_bpp: float
    h_after_bpp: float
    loss_bpp: float
    frac_pixels_affected: float
    n_pixels_measured: int


@dataclass(frozen=True)
class ChromaMetrics:
    """Chroma-related metrics for one image at one ``y1`` value.

    Runtime Attributes
    ------------------
    - ``y1`` (float): Low-luminance threshold or tested Y value.
    - ``affected_fraction`` (float): Fraction of pixels affected, in [0, 1].
    - ``entropy_before_bpp`` (float): Chroma entropy before treatment, in bits
      per pixel.
    - ``entropy_after_bpp`` (float): Chroma entropy after treatment, in bits per
      pixel.
    - ``rel_entropy_loss_pct`` (float): Relative entropy loss, in percent.
    - ``mean_chroma_before`` (float): Mean chroma before treatment.
    - ``mean_chroma_after`` (float): Mean chroma after treatment.
    - ``rel_mean_chroma_loss_pct`` (float): Relative mean chroma loss, in
      percent.
    - ``delta_e76_mean`` (float): Mean CIE76 color difference.
    - ``delta_e76_p95`` (float): 95th percentile CIE76 color difference.
    - ``n_pixels_measured`` (int): Number of pixels included in the measurement.
    """

    y1: float
    affected_fraction: float  # in [0,1]
    entropy_before_bpp: float
    entropy_after_bpp: float
    rel_entropy_loss_pct: float  # percent
    mean_chroma_before: float
    mean_chroma_after: float
    rel_mean_chroma_loss_pct: float  # percent
    delta_e76_mean: float
    delta_e76_p95: float
    n_pixels_measured: int


@dataclass(frozen=True)
class AggregateMetric:
    """Mean and standard deviation summary for a metric across images.

    Runtime Attributes
    ------------------
    - ``mean`` (float): Mean value across images.
    - ``std`` (float): Standard deviation across images.
    """

    mean: float
    std: float


@dataclass(frozen=True)
class AggregateRow:
    """Aggregate chroma metrics for one ``y1`` value.

    Runtime Attributes
    ------------------
    - ``y1`` (float): Low-luminance threshold or tested Y value.
    - ``affected_pct`` (AggregateMetric): Percentage of affected pixels.
    - ``entropy_before_bpp`` (AggregateMetric): Entropy before treatment, in bits
      per pixel.
    - ``entropy_after_bpp`` (AggregateMetric): Entropy after treatment, in bits
      per pixel.
    - ``rel_entropy_loss_pct`` (AggregateMetric): Relative entropy loss, in
      percent.
    - ``rel_mean_chroma_loss_pct`` (AggregateMetric): Relative mean chroma loss,
      in percent.
    - ``delta_e76_mean`` (AggregateMetric): Mean CIE76 color difference.
    - ``delta_e76_p95`` (AggregateMetric): 95th percentile CIE76 color
      difference.
    """

    y1: float
    affected_pct: AggregateMetric
    entropy_before_bpp: AggregateMetric
    entropy_after_bpp: AggregateMetric
    rel_entropy_loss_pct: AggregateMetric
    rel_mean_chroma_loss_pct: AggregateMetric
    delta_e76_mean: AggregateMetric
    delta_e76_p95: AggregateMetric


@dataclass(frozen=True)
class ChromaInfoRetention:
    """Information-theoretic chroma retention statistics.

    Runtime Attributes
    ------------------
    - ``h_before_bpp`` (float): Chroma entropy before treatment, in bits per
      pixel.
    - ``h_after_bpp`` (float): Chroma entropy after treatment, in bits per
      pixel.
    - ``mi_bpp`` (float): Mutual information estimate, in bits per pixel.
    - ``cir`` (float): Chroma information retention in [0, 1].
    """
    h_before_bpp: float
    h_after_bpp: float
    mi_bpp: float
    cir: float  # chroma information retention in [0,1]


# =============================================================================
# Helpers: validation and stats
# =============================================================================

def _entropy_bits_from_counts(counts: np.ndarray) -> float:
    """Compute Shannon entropy (bits) from histogram counts."""
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def _quantize_ab(
    a: np.ndarray,
    b: np.ndarray,
    *,
    nbits_per_axis: int,
    ab_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Quantize Lab a*,b* into integer bins.

    Parameters
    ----------        a: Array of a* values.
        b: Array of b* values.
        nbits_per_axis: Bits per axis (e.g., 8 -> 256 bins per axis).
        ab_range: (min,max) clipping range for a* and b*.

    Returns
    -------        (a_q, b_q, bins) with a_q,b_q in [0, bins-1].
    """
    a_min, a_max = ab_range
    bins = int(2 ** int(nbits_per_axis))

    a_c = np.clip(a, a_min, a_max)
    b_c = np.clip(b, a_min, a_max)

    # Map range -> [0, bins)
    scale = (bins - 1e-12) / (a_max - a_min)
    a_q = np.floor((a_c - a_min) * scale).astype(np.int64)
    b_q = np.floor((b_c - a_min) * scale).astype(np.int64)

    a_q = np.clip(a_q, 0, bins - 1)
    b_q = np.clip(b_q, 0, bins - 1)
    return a_q, b_q, bins


def relative_mean_chroma_loss_pct_global_lab(
    *,
    converter: ColorTreatment,
    srgb_before_01: np.ndarray,
    srgb_after_01: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """Compute global relative mean chroma loss in CIE Lab.

    The metric is:
        100 * (E[C*_before] - E[C*_after]) / E[C*_before]
    where:
        C* = sqrt(a*^2 + b*^2)

    Parameters
    ----------        converter: ColorTreatment instance (used for sRGB->Lab).
        srgb_before_01: Original sRGB image in [0,1], shape (H, W, 3).
        srgb_after_01: Processed sRGB image in [0,1], shape (H, W, 3).
        eps: Small constant to avoid division by zero.

    Returns
    -------        A tuple: (loss_pct, mean_c_before, mean_c_after).
    """
    lab0 = converter.sRGB_to_lab(srgb_before_01)
    lab1 = converter.sRGB_to_lab(srgb_after_01)

    c0 = np.sqrt(lab0[..., 1] ** 2 + lab0[..., 2] ** 2)
    c1 = np.sqrt(lab1[..., 1] ** 2 + lab1[..., 2] ** 2)

    mean_c0 = float(np.mean(c0))
    mean_c1 = float(np.mean(c1))

    loss_pct = 100.0 * (mean_c0 - mean_c1) / max(mean_c0, eps)
    return float(loss_pct), mean_c0, mean_c1


def chroma_information_retention_lab_ab(
    converter: ColorTreatment,
    srgb_before_01: np.ndarray,
    srgb_after_01: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    nbits_per_axis: int = 8,
    ab_range: Tuple[float, float] = (-128.0, 127.0),
) -> ChromaInfoRetention:
    """Compute chroma information retention (CIR) using sparse MI on quantized Lab (a*,b*)."""
    lab0 = converter.sRGB_to_lab(srgb_before_01)
    lab1 = converter.sRGB_to_lab(srgb_after_01)

    a0, b0 = lab0[..., 1], lab0[..., 2]
    a1, b1 = lab1[..., 1], lab1[..., 2]

    if mask is not None:
        if mask.shape != a0.shape:
            raise ValueError(f"mask shape {mask.shape} must match image spatial shape {a0.shape}")
        a0, b0, a1, b1 = a0[mask], b0[mask], a1[mask], b1[mask]

    if a0.size == 0:
        return ChromaInfoRetention(0.0, 0.0, 0.0, 0.0)

    a0_q, b0_q, bins = _quantize_ab(a0, b0, nbits_per_axis=nbits_per_axis, ab_range=ab_range)
    a1_q, b1_q, _ = _quantize_ab(a1, b1, nbits_per_axis=nbits_per_axis, ab_range=ab_range)

    x = (a0_q * bins + b0_q).astype(np.int64).ravel()
    y = (a1_q * bins + b1_q).astype(np.int64).ravel()
    n_states = int(bins * bins)

    # Now bincount is happy (1D)
    px = np.bincount(x, minlength=n_states).astype(np.float64)
    py = np.bincount(y, minlength=n_states).astype(np.float64)

    h_x = _entropy_bits_from_counts(px)
    h_y = _entropy_bits_from_counts(py)

    if h_x <= 0.0:
        return ChromaInfoRetention(h_x, h_y, 0.0, 0.0)

    n = float(x.size)
    px /= n
    py /= n

    # Sparse joint: only observed (x,y) pairs
    key = (x * n_states + y).astype(np.int64)  # x,y are 1D now
    uniq_key, joint_counts = np.unique(key, return_counts=True)

    pxy = joint_counts.astype(np.float64) / n
    x_u = (uniq_key // n_states).astype(np.int64)
    y_u = (uniq_key % n_states).astype(np.int64)

    denom = px[x_u] * py[y_u]
    # safe: denom > 0 by construction, but guard anyway
    valid = (pxy > 0.0) & (denom > 0.0)

    mi = float(np.sum(pxy[valid] * np.log2(pxy[valid] / denom[valid])))

    cir = float(mi / h_x)
    cir = float(np.clip(cir, 0.0, 1.0))

    return ChromaInfoRetention(h_before_bpp=h_x, h_after_bpp=h_y, mi_bpp=mi, cir=cir)


def _validate_srgb_01(srgb: np.ndarray) -> np.ndarray:
    """Validate and return sRGB as float64 in [0,1].

    This function is intentionally strict to avoid silent bugs (e.g., uint8 [0..255]
    being clipped to [0..1]).

    Parameters
    ----------        srgb: Image array of shape (H, W, 3). Expected range [0, 1].

    Returns
    -------        Float64 sRGB in [0, 1].

    Raises
    ------        ValueError: If shape is wrong or values are outside [0,1] by a non-trivial margin.
    """
    if srgb.ndim != 3 or srgb.shape[-1] != 3:
        raise ValueError(f"Expected sRGB image shape (H,W,3). Got {srgb.shape}")

    if srgb.dtype.kind not in ("f", "u", "i"):
        raise ValueError(f"Expected numeric image dtype. Got {srgb.dtype}")

    srgb_f = srgb.astype(np.float64, copy=False)
    mx = float(np.max(srgb_f))
    mn = float(np.min(srgb_f))

    # Allow tiny numeric excursions from previous computations.
    eps = 1e-6
    if mx > 1.0 + eps or mn < 0.0 - eps:
        raise ValueError(
            f"Expected sRGB in [0,1], got range [{mn:.4g}, {mx:.4g}]. "
            "If your image is 8-bit, convert with: srgb = srgb_uint8.astype(float)/255."
        )

    return np.clip(srgb_f, 0.0, 1.0)


def _mean_std(values: Sequence[float]) -> AggregateMetric:
    """Compute mean and std (ddof=0) for a sequence of floats."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return AggregateMetric(mean=float("nan"), std=float("nan"))
    return AggregateMetric(mean=float(np.mean(arr)), std=float(np.std(arr)))


def _percent(v: float) -> float:
    """Convert fraction [0,1] to percent."""
    return 100.0 * v


# =============================================================================
# Entropy metric (joint histogram of Lab a*,b*)
# =============================================================================

def _shannon_entropy_bits_from_counts(counts: np.ndarray) -> float:
    """Compute Shannon entropy (bits) from histogram counts."""
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0

    p = counts.astype(np.float64) / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def _joint_entropy_lab_ab_bpp(
    converter: ColorTreatment,
    srgb_01: np.ndarray,
    mask: Optional[np.ndarray],
    nbits_per_axis: int,
    ab_range: Tuple[float, float],
) -> Tuple[float, int]:
    """Compute joint entropy H(a*,b*) in bits/pixel for an sRGB image.

    Parameters
    ----------        converter: ColorTreatment instance.
        srgb_01: sRGB image in [0,1], shape (H,W,3), float.
        mask: Optional boolean mask (H,W). If provided, only masked pixels are measured.
        nbits_per_axis: Quantization bits per axis for a* and b* (e.g., 8 -> 256 bins).
        ab_range: Clipping range for a* and b* before quantization.

    Returns
    -------        Tuple of (entropy_bits_per_pixel, n_pixels_used).
    """
    lab = converter.sRGB_to_lab(srgb_01)
    a = lab[..., 1]
    b = lab[..., 2]

    if mask is not None:
        if mask.shape != a.shape:
            raise ValueError(f"mask shape {mask.shape} must match image spatial shape {a.shape}")
        a = a[mask]
        b = b[mask]

    n = int(a.size)
    if n == 0:
        return 0.0, 0

    a_min, a_max = ab_range
    bins = int(2 ** int(nbits_per_axis))

    # Clip then quantize to bins [0..bins-1]
    a_c = np.clip(a, a_min, a_max)
    b_c = np.clip(b, a_min, a_max)

    # Map range -> [0, bins)
    # Use (bins - 1e-12) to avoid mapping exactly max to bins (out of range).
    scale = (bins - 1e-12) / (a_max - a_min)
    a_q = np.floor((a_c - a_min) * scale).astype(np.int64)
    b_q = np.floor((b_c - a_min) * scale).astype(np.int64)
    a_q = np.clip(a_q, 0, bins - 1)
    b_q = np.clip(b_q, 0, bins - 1)

    # Joint histogram
    counts = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(counts, (a_q, b_q), 1)

    h = _shannon_entropy_bits_from_counts(counts)
    return h, n


# =============================================================================
# Core pipeline: apply desaturation and compute metrics
# =============================================================================

def _delta_e76_stats(
    converter: ColorTreatment,
    srgb_before: np.ndarray,
    srgb_after: np.ndarray,
    mask: Optional[np.ndarray],
) -> Tuple[float, float]:
    """Compute ΔE76 mean and p95 between two sRGB images, optionally within a mask.

    Parameters
    ----------        converter: ColorTreatment instance.
        srgb_before: sRGB image in [0,1], shape (H,W,3).
        srgb_after: sRGB image in [0,1], shape (H,W,3).
        mask: Optional boolean mask (H,W).

    Returns
    -------        (mean_delta_e76, p95_delta_e76)
    """
    lab0 = converter.sRGB_to_lab(srgb_before)
    lab1 = converter.sRGB_to_lab(srgb_after)
    de = np.sqrt(np.sum((lab0 - lab1) ** 2, axis=-1))

    if mask is not None:
        de = de[mask]

    if de.size == 0:
        return 0.0, 0.0

    return float(np.mean(de)), float(np.quantile(de, 0.95))


def _mean_chroma(
    converter: ColorTreatment,
    srgb: np.ndarray,
    mask: Optional[np.ndarray],
) -> float:
    """Compute mean Lab chroma C* = sqrt(a*^2 + b*^2), optionally within a mask."""
    lab = converter.sRGB_to_lab(srgb)
    c = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    if mask is not None:
        c = c[mask]
    if c.size == 0:
        return 0.0
    return float(np.mean(c))


def compute_chroma_metrics_for_image(
    srgb_01: np.ndarray,
    *,
    y1: float,
    rec_standard: REC_STANDARD = "rec709",
    nbits_per_axis: int = 8,
    ab_range: Tuple[float, float] = (-128.0, 127.0),
    measure_mask: Optional[np.ndarray] = None,
    shadow_only: bool = False,
) -> ChromaMetrics:
    """Compute chroma-loss metrics for a single image at one ``y1`` value.

    Strategy: desaturate xy toward the D65 white point using a smoothstep strength s(Y_orig)
    applied on low luminance only, with Y_orig computed from the ORIGINAL image.

    Metrics:
      - affected_fraction
      - joint chroma entropy (Lab a*,b*) before/after, and relative loss
      - mean chroma C* before/after, and relative loss
      - ΔE76 mean and p95 (optionally within a measurement mask)

    Parameters
    ----------
    srgb_01 : np.ndarray
        sRGB image in [0, 1], shape (H, W, 3).

    y1 : float
        Threshold controlling the low-luminance desaturation range.

    rec_standard : REC_STANDARD
        Color standard: ``"rec601"``, ``"rec709"``, or ``"rec2020"``.

    nbits_per_axis : int
        Quantization bits per Lab a* and b* axis for joint entropy.

    ab_range : Tuple[float, float]
        Clipping range for a* and b* before quantization.

    measure_mask : Optional[np.ndarray]
        Optional boolean mask with shape (H, W) restricting all measurements.

    shadow_only : bool
        If True, further restricts measurements to pixels where ``Y_orig < y1``.

    Returns
    -------
    ChromaMetrics
        Chroma metrics for the image and threshold.
    """
    srgb = _validate_srgb_01(srgb_01)
    converter = ColorTreatment(rec_standard=rec_standard)

    # ORIGINAL xyY and Y_orig (must be original luminance proxy)
    xyY = converter.sRGB_to_xyY(srgb)
    y_orig = xyY[..., 2]
    xy = xyY[..., :2]

    y1f = float(y1)

    # Strength map and affected fraction
    s_map = ColorTreatment.desat_strength_low_Yorig(y_orig, Y1=y1f)
    affected = (s_map > 0.0)

    # Compose effective measurement mask
    mask_eff: Optional[np.ndarray]
    if measure_mask is None:
        mask_eff = affected if shadow_only else None
    else:
        if measure_mask.shape != y_orig.shape:
            raise ValueError(f"measure_mask shape {measure_mask.shape} must match image spatial shape {y_orig.shape}")
        if shadow_only:
            mask_eff = measure_mask & affected
        else:
            mask_eff = measure_mask

    if mask_eff is None:
        affected_fraction = float(np.mean(affected))
    else:
        denom = int(np.sum(mask_eff))
        affected_fraction = float(np.mean(affected[mask_eff])) if denom > 0 else 0.0

    # Apply your desaturation strategy (xy -> white) using Y_orig
    xy_new = ColorTreatment.desaturate_xyY_toward_white_low(
        Y_orig=y_orig,
        xy=xy,
        Y1=y1f,
    )

    # Recompose xyY with ORIGINAL Y (preserve Y)
    xyY_new = np.empty_like(xyY)
    xyY_new[..., 0] = xy_new[..., 0]
    xyY_new[..., 1] = xy_new[..., 1]
    xyY_new[..., 2] = y_orig

    # Back to sRGB through your pipeline
    srgb_new = converter.xyY_to_sRGB(xyY_new)
    srgb_new = np.clip(srgb_new, 0.0, 1.0)

    # Entropy before/after
    h_before, n_before = _joint_entropy_lab_ab_bpp(
        converter=converter,
        srgb_01=srgb,
        mask=mask_eff,
        nbits_per_axis=nbits_per_axis,
        ab_range=ab_range,
    )
    h_after, n_after = _joint_entropy_lab_ab_bpp(
        converter=converter,
        srgb_01=srgb_new,
        mask=mask_eff,
        nbits_per_axis=nbits_per_axis,
        ab_range=ab_range,
    )

    rel_entropy_loss_pct = 0.0 if h_before <= 0.0 else float((h_before - h_after) / h_before * 100.0)

    # Mean chroma before/after
    c_before = _mean_chroma(converter, srgb, mask_eff)
    c_after = _mean_chroma(converter, srgb_new, mask_eff)
    rel_mean_chroma_loss_pct = 0.0 if c_before <= 0.0 else float((c_before - c_after) / c_before * 100.0)

    # ΔE76 stats
    de_mean, de_p95 = _delta_e76_stats(converter, srgb, srgb_new, mask_eff)

    return ChromaMetrics(
        y1=y1f,
        affected_fraction=affected_fraction,
        entropy_before_bpp=float(h_before),
        entropy_after_bpp=float(h_after),
        rel_entropy_loss_pct=float(rel_entropy_loss_pct),
        mean_chroma_before=float(c_before),
        mean_chroma_after=float(c_after),
        rel_mean_chroma_loss_pct=float(rel_mean_chroma_loss_pct),
        delta_e76_mean=float(de_mean),
        delta_e76_p95=float(de_p95),
        n_pixels_measured=int(n_after),
    )


def aggregate_chroma_metrics(
    images_srgb_01: Sequence[np.ndarray],
    *,
    y1_values: Union[np.ndarray, Iterable[float]],
    rec_standard: REC_STANDARD = "rec709",
    nbits_per_axis: int = 8,
    ab_range: Tuple[float, float] = (-128.0, 127.0),
    measure_mask: Optional[np.ndarray] = None,
    shadow_only: bool = False,
) -> List[AggregateRow]:
    """Aggregate chroma-loss metrics across a list of images.

    Parameters
    ----------
    images_srgb_01 : Sequence[np.ndarray]
        Sequence of sRGB images in [0, 1], each with shape (H, W, 3).

    y1_values : Union[np.ndarray, Iterable[float]]
        Iterable of low-luminance thresholds.

    rec_standard : REC_STANDARD
        Color standard: ``"rec601"``, ``"rec709"``, or ``"rec2020"``.

    nbits_per_axis : int
        Quantization bits per Lab a* and b* axis for entropy.

    ab_range : Tuple[float, float]
        Clipping range for a* and b* before quantization.

    measure_mask : Optional[np.ndarray]
        Optional boolean mask with shape (H, W) applied to every image.

    shadow_only : bool
        If True, measurements are restricted to pixels where ``Y_orig < y1``.

    Returns
    -------
    List[AggregateRow]
        One aggregate row per threshold.
    """
    if len(images_srgb_01) == 0:
        raise ValueError("images_srgb_01 is empty.")

    rows: List[AggregateRow] = []
    for y1 in y1_values:
        metrics = [
            compute_chroma_metrics_for_image(
                img,
                y1=float(y1),
                rec_standard=rec_standard,
                nbits_per_axis=nbits_per_axis,
                ab_range=ab_range,
                measure_mask=measure_mask,
                shadow_only=shadow_only,
            )
            for img in images_srgb_01
        ]

        rows.append(
            AggregateRow(
                y1=float(y1),
                affected_pct=_mean_std([_percent(m.affected_fraction) for m in metrics]),
                entropy_before_bpp=_mean_std([m.entropy_before_bpp for m in metrics]),
                entropy_after_bpp=_mean_std([m.entropy_after_bpp for m in metrics]),
                rel_entropy_loss_pct=_mean_std([m.rel_entropy_loss_pct for m in metrics]),
                rel_mean_chroma_loss_pct=_mean_std([m.rel_mean_chroma_loss_pct for m in metrics]),
                delta_e76_mean=_mean_std([m.delta_e76_mean for m in metrics]),
                delta_e76_p95=_mean_std([m.delta_e76_p95 for m in metrics]),
            )
        )

    return rows


# =============================================================================
# Reporting: plots + Markdown report (English), saved under assets/graphics
# =============================================================================

def _default_graphics_dir() -> Path:
    """Return shinier/color/assets/graphics path (relative to this module)."""
    return Path(__file__).resolve().parent / "assets" / "graphics"


def _default_report_dir() -> Path:
    """Return shinier/color/assets/reports path (relative to this module)."""
    return Path(__file__).resolve().parent / "assets"


def _ensure_dir(path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def _format_mean_std(metric: AggregateMetric, width: int = 5, decimals: int = 2) -> str:
    """Format mean ± std with a fixed width for Markdown tables."""
    fmt = f"{{:>{width}.{decimals}f}}"
    return f"{fmt.format(metric.mean)}±{fmt.format(metric.std)}"


def _save_errorbar_plot(
    *,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Save an errorbar plot using matplotlib (imported lazily)."""
    import matplotlib.pyplot as plt  # local import to keep module lightweight unless used

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_chroma_loss_report(
    images_srgb_01: Sequence[np.ndarray],
    *,
    y1_values: Union[np.ndarray, Iterable[float]],
    rec_standard: REC_STANDARD = "rec709",
    nbits_per_axis: int = 8,
    ab_range: Tuple[float, float] = (-128.0, 127.0),
    graphics_dir: Optional[Union[str, Path]] = None,
    report_dir: Optional[Union[str, Path]] = None,
    report_filename: str = "chroma_loss_report.md",
) -> Path:
    """Generate a Markdown report and plots for chroma-loss vs ``y1``.

    This function creates:
      - Aggregate report (mean ± std across images)
      - Shadow-only report (measurement restricted to Y_orig < Y1)
      - Plots saved under: shinier/color/assets/graphics/ (by default)

    Parameters
    ----------
    images_srgb_01 : Sequence[np.ndarray]
        Sequence of sRGB images in [0, 1], each with shape (H, W, 3).

    y1_values : Union[np.ndarray, Iterable[float]]
        Iterable of low-luminance thresholds.

    rec_standard : REC_STANDARD
        Color standard: ``"rec601"``, ``"rec709"``, or ``"rec2020"``.

    nbits_per_axis : int
        Quantization bits per Lab a* and b* axis for entropy.

    ab_range : Tuple[float, float]
        Clipping range for a* and b* before quantization.

    graphics_dir : Optional[Union[str, Path]]
        Output directory for plots. If None, uses the package graphics folder.

    report_dir : Optional[Union[str, Path]]
        Output directory for the Markdown report. If None, uses the package
        assets folder.

    report_filename : str
        Name of the Markdown report file.

    Returns
    -------
    Path
        Path to the generated Markdown report.
    """
    out_dir = _default_graphics_dir() if graphics_dir is None else Path(graphics_dir)
    _ensure_dir(out_dir)
    report_dir = _default_report_dir() if report_dir is None else Path(report_dir)
    _ensure_dir(report_dir)

    # Aggregate + shadow-only tables
    agg_rows = aggregate_chroma_metrics(
        images_srgb_01,
        y1_values=y1_values,
        rec_standard=rec_standard,
        nbits_per_axis=nbits_per_axis,
        ab_range=ab_range,
        shadow_only=False,
    )
    sh_rows = aggregate_chroma_metrics(
        images_srgb_01,
        y1_values=y1_values,
        rec_standard=rec_standard,
        nbits_per_axis=nbits_per_axis,
        ab_range=ab_range,
        shadow_only=True,
    )

    # Build arrays for plots
    y = np.array([r.y1 for r in agg_rows], dtype=np.float64)

    def arr(metric_name: str, rows: List[AggregateRow]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mean and standard-deviation vectors for one aggregate metric."""
        m = np.array([getattr(r, metric_name).mean for r in rows], dtype=np.float64)
        s = np.array([getattr(r, metric_name).std for r in rows], dtype=np.float64)
        return m, s

    # Plots (aggregate)
    affected_mean, affected_std = arr("affected_pct", agg_rows)
    rel_ent_mean, rel_ent_std = arr("rel_entropy_loss_pct", agg_rows)
    rel_c_mean, rel_c_std = arr("rel_mean_chroma_loss_pct", agg_rows)
    de_mean, de_std = arr("delta_e76_mean", agg_rows)
    de_p95_mean, de_p95_std = arr("delta_e76_p95", agg_rows)

    p_agg_affected = out_dir / "aggregate_affected_vs_y1.png"
    p_agg_rel_ent = out_dir / "aggregate_rel_entropy_loss_vs_y1.png"
    p_agg_de_mean = out_dir / "aggregate_de_mean_vs_y1.png"
    p_agg_de_p95 = out_dir / "aggregate_de_p95_vs_y1.png"
    p_agg_rel_c = out_dir / "aggregate_rel_mean_chroma_loss_vs_y1.png"

    _save_errorbar_plot(
        x=y, y=affected_mean, yerr=affected_std,
        title="Pixels affected vs Y1 (aggregate)",
        xlabel="Y1",
        ylabel="Affected pixels (%)",
        out_path=p_agg_affected,
    )
    _save_errorbar_plot(
        x=y, y=rel_ent_mean, yerr=rel_ent_std,
        title="Relative chroma entropy loss vs Y1 (aggregate)",
        xlabel="Y1",
        ylabel="Relative entropy loss (%)",
        out_path=p_agg_rel_ent,
    )
    _save_errorbar_plot(
        x=y, y=de_mean, yerr=de_std,
        title="ΔE76 mean vs Y1 (aggregate)",
        xlabel="Y1",
        ylabel="ΔE76 mean",
        out_path=p_agg_de_mean,
    )
    _save_errorbar_plot(
        x=y, y=de_p95_mean, yerr=de_p95_std,
        title="ΔE76 p95 vs Y1 (aggregate)",
        xlabel="Y1",
        ylabel="ΔE76 p95",
        out_path=p_agg_de_p95,
    )
    _save_errorbar_plot(
        x=y, y=rel_c_mean, yerr=rel_c_std,
        title="Relative mean chroma loss vs Y1 (aggregate)",
        xlabel="Y1",
        ylabel="Relative mean chroma loss (%)",
        out_path=p_agg_rel_c,
    )

    # Plots (shadow-only)
    sh_rel_ent_mean, sh_rel_ent_std = arr("rel_entropy_loss_pct", sh_rows)
    sh_rel_c_mean, sh_rel_c_std = arr("rel_mean_chroma_loss_pct", sh_rows)
    sh_de_mean, sh_de_std = arr("delta_e76_mean", sh_rows)
    sh_de_p95_mean, sh_de_p95_std = arr("delta_e76_p95", sh_rows)

    p_sh_rel_ent = out_dir / "shadow_rel_entropy_loss_vs_y1.png"
    p_sh_rel_c = out_dir / "shadow_rel_mean_chroma_loss_vs_y1.png"
    p_sh_de_mean = out_dir / "shadow_de_mean_vs_y1.png"
    p_sh_de_p95 = out_dir / "shadow_de_p95_vs_y1.png"

    _save_errorbar_plot(
        x=y, y=sh_rel_ent_mean, yerr=sh_rel_ent_std,
        title="Relative chroma entropy loss vs Y1 (shadow-only region)",
        xlabel="Y1",
        ylabel="Relative entropy loss (%)",
        out_path=p_sh_rel_ent,
    )
    _save_errorbar_plot(
        x=y, y=sh_rel_c_mean, yerr=sh_rel_c_std,
        title="Relative mean chroma loss vs Y1 (shadow-only region)",
        xlabel="Y1",
        ylabel="Relative mean chroma loss (%)",
        out_path=p_sh_rel_c,
    )
    _save_errorbar_plot(
        x=y, y=sh_de_mean, yerr=sh_de_std,
        title="ΔE76 mean vs Y1 (shadow-only region)",
        xlabel="Y1",
        ylabel="ΔE76 mean",
        out_path=p_sh_de_mean,
    )
    _save_errorbar_plot(
        x=y, y=sh_de_p95_mean, yerr=sh_de_p95_std,
        title="ΔE76 p95 vs Y1 (shadow-only region)",
        xlabel="Y1",
        ylabel="ΔE76 p95",
        out_path=p_sh_de_p95,
    )

    # Markdown report (English)
    rel = lambda p: p.name  # keep links relative within the graphics directory

    def md_table(rows: List[AggregateRow]) -> str:
        """Render aggregated chroma metrics as a Markdown table."""
        header = (
            "| Y1 | Affected% | Entropy before/after (bpp) | Rel entropy loss% | "
            "Rel mean chroma loss% | ΔE76 mean | ΔE76 p95 |\n"
            "|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        lines = []
        for r in rows:
            lines.append(
                f"| {r.y1:0.3f} | {_format_mean_std(r.affected_pct)} | "
                f"{_format_mean_std(r.entropy_before_bpp, decimals=3)}/"
                f"{_format_mean_std(r.entropy_after_bpp, decimals=3)} | "
                f"{_format_mean_std(r.rel_entropy_loss_pct)} | "
                f"{_format_mean_std(r.rel_mean_chroma_loss_pct)} | "
                f"{_format_mean_std(r.delta_e76_mean, decimals=3)} | "
                f"{_format_mean_std(r.delta_e76_p95, decimals=3)} |"
            )
        return header + "\n".join(lines) + "\n"

    report_path = report_dir / report_filename

    from textwrap import dedent

    n_images = len(images_srgb_01)
    agg_table_md = md_table(agg_rows)
    sh_table_md = md_table(sh_rows)

    agg_affected_link = rel(p_agg_affected)
    agg_rel_ent_link = rel(p_agg_rel_ent)
    agg_de_mean_link = rel(p_agg_de_mean)
    agg_de_p95_link = rel(p_agg_de_p95)
    agg_rel_c_link = rel(p_agg_rel_c)

    sh_rel_ent_link = rel(p_sh_rel_ent)
    sh_rel_c_link = rel(p_sh_rel_c)
    sh_de_mean_link = rel(p_sh_de_mean)
    sh_de_p95_link = rel(p_sh_de_p95)

    REPORT_TEMPLATE = dedent("""\
    # Chroma-loss report — low-luminance desaturation toward white (xyY)

    This report summarizes a test over **{n_images} images** (mean ± std), for a chroma-stabilization strategy:

    - Working space: **xyY**
    - Operation: desaturate **(x, y)** toward the **D65 white point**
    - Control signal: **original luminance** `Y_orig` (computed from the original image)
    - Strength: a **smoothstep** ramp on low luminance only, with a single parameter **`Y1`**
      - `s(Y_orig)=1` at `Y_orig=0` (full pull to white)
      - `s(Y_orig)=0` for `Y_orig>=Y1` (no desaturation)

    ## Metrics (plain-language interpretation)

    - **Affected%**: fraction of pixels with `s(Y_orig) > 0` (i.e., pixels in the low-luminance ramp).

    - **Entropy (bpp)**: Shannon entropy of the joint histogram of **Lab (a\\*, b\\*)** (bits/pixel).  
      Higher values indicate a richer or more evenly distributed set of chroma states.

    - **Rel entropy loss% (Lab a\\*, b\\*)**: Relative drop in chroma diversity or complexity.  
      We quantize the chroma plane $(a^*, b^*)$ and compute the Shannon entropy of its 2-D histogram (bits per pixel).  
      A higher entropy means the image uses a wider and/or more evenly populated set of chroma states.  
      A decrease means the chroma distribution becomes more concentrated (fewer distinct chroma states or less even usage), even if the average saturation stays similar.

    - **Rel mean chroma loss% (C\\*)**: Relative drop in average saturation strength.  
      Chroma magnitude is computed per pixel as

      $$
      C^* = \\sqrt{{a^{{*2}} + b^{{*2}}}}
      $$

      We then report the relative change of the mean value of $C^*$.  
      This captures how much the image becomes less saturated on average, regardless of whether the remaining colors are still diverse.

    - **ΔE\\*76 mean / p95**: Magnitude of the actual color change in CIE Lab.  
      ΔE\\*76 (often written ΔE76) is the Euclidean distance between two colors in Lab:

      $$
      \\Delta E_{{76}} = \\sqrt{{(\\Delta L^*)^2 + (\\Delta a^*)^2 + (\\Delta b^*)^2}}
      $$

      The “76” refers to the original 1976 CIE L\\*a\\*b\\* formulation where this distance metric was introduced.  
      We report the mean ΔE76 (average change) and the 95th percentile (how large the change is for the worst 5% of pixels).

    > Rough ΔE76 landmarks (very approximate):  
    > ~0–1 imperceptible, ~1–2 subtle, ~2–4 visible, >5 strong.
    ---

    ## Aggregate results (full image; mean ± std across images)

    {agg_table_md}

    ## Shadow-only results (measurement region = Y_orig < Y1)

    {sh_table_md}

    ---

    ## Plots — Aggregate (full image)

    ### Pixels affected vs Y1
    ![]({agg_affected_link})

    ### Relative chroma entropy loss vs Y1
    ![]({agg_rel_ent_link})

    ### ΔE76 mean vs Y1
    ![]({agg_de_mean_link})

    ### ΔE76 p95 vs Y1
    ![]({agg_de_p95_link})

    ### Relative mean chroma loss vs Y1
    ![]({agg_rel_c_link})

    ---

    ## Plots — Shadow-only (Y_orig < Y1)

    ### Relative chroma entropy loss vs Y1
    ![]({sh_rel_ent_link})

    ### Relative mean chroma loss vs Y1
    ![]({sh_rel_c_link})

    ### ΔE76 mean vs Y1
    ![]({sh_de_mean_link})

    ### ΔE76 p95 vs Y1
    ![]({sh_de_p95_link})
    """)

    report_text = REPORT_TEMPLATE.format(
        n_images=n_images,
        agg_table_md=agg_table_md,
        sh_table_md=sh_table_md,
        agg_affected_link=agg_affected_link,
        agg_rel_ent_link=agg_rel_ent_link,
        agg_de_mean_link=agg_de_mean_link,
        agg_de_p95_link=agg_de_p95_link,
        agg_rel_c_link=agg_rel_c_link,
        sh_rel_ent_link=sh_rel_ent_link,
        sh_rel_c_link=sh_rel_c_link,
        sh_de_mean_link=sh_de_mean_link,
        sh_de_p95_link=sh_de_p95_link,
    )

    report_path.write_text(report_text, encoding="utf-8")
    return report_path


# =============================================================================
# Backward-compatible function (entropy-only)
# =============================================================================

def chroma_info_loss_bits_per_pixel_vs_y1(
    srgb_01: np.ndarray,
    *,
    rec_standard: REC_STANDARD = "rec709",
    y1_values: Union[np.ndarray, Iterable[float]] = (0.02, 0.05, 0.08, 0.10, 0.12),
    nbits_per_axis: int = 8,
    ab_range: Tuple[float, float] = (-128.0, 127.0),
    measure_mask: Optional[np.ndarray] = None,
    low_y_only: bool = False,
) -> List[ChromaInfoLossResult]:
    """Compute legacy entropy-only chroma information loss vs ``y1``.

    Notes
    -----
    - Input must be float in [0, 1]. This function is strict and raises on
      uint8-style [0, 255] input.
    - For richer metrics, use ``generate_chroma_loss_report``.
    """
    srgb = _validate_srgb_01(srgb_01)

    converter = ColorTreatment(rec_standard=rec_standard)

    xyY = converter.sRGB_to_xyY(srgb)
    y_orig = xyY[..., 2]
    xy = xyY[..., :2]

    results: List[ChromaInfoLossResult] = []

    if not low_y_only:
        h_before, n_before = _joint_entropy_lab_ab_bpp(
            converter=converter,
            srgb_01=srgb,
            mask=measure_mask,
            nbits_per_axis=nbits_per_axis,
            ab_range=ab_range,
        )

    for y1 in y1_values:
        y1f = float(y1)

        xy_new = ColorTreatment.desaturate_xyY_toward_white_low(
            Y_orig=y_orig,
            xy=xy,
            Y1=y1f,
        )

        xyY_new = np.empty_like(xyY)
        xyY_new[..., 0] = xy_new[..., 0]
        xyY_new[..., 1] = xy_new[..., 1]
        xyY_new[..., 2] = y_orig

        srgb_new = converter.xyY_to_sRGB(xyY_new)
        srgb_new = np.clip(srgb_new, 0.0, 1.0)

        if low_y_only:
            low_mask = (y_orig < y1f)
            if measure_mask is None:
                mask_eff = low_mask
            else:
                mask_eff = measure_mask & low_mask

            h_before_eff, n_eff = _joint_entropy_lab_ab_bpp(
                converter=converter,
                srgb_01=srgb,
                mask=mask_eff,
                nbits_per_axis=nbits_per_axis,
                ab_range=ab_range,
            )
        else:
            mask_eff = measure_mask
            h_before_eff = h_before
            n_eff = n_before

        h_after, n_after = _joint_entropy_lab_ab_bpp(
            converter=converter,
            srgb_01=srgb_new,
            mask=mask_eff,
            nbits_per_axis=nbits_per_axis,
            ab_range=ab_range,
        )

        s_map = ColorTreatment.desat_strength_low_Yorig(y_orig, Y1=y1f)
        affected = (s_map > 0.0)
        if mask_eff is None:
            frac_affected = float(np.mean(affected))
        else:
            denom = int(np.sum(mask_eff))
            frac_affected = float(np.mean(affected[mask_eff])) if denom > 0 else 0.0

        results.append(
            ChromaInfoLossResult(
                y1=y1f,
                h_before_bpp=float(h_before_eff),
                h_after_bpp=float(h_after),
                loss_bpp=float(h_before_eff - h_after),
                frac_pixels_affected=frac_affected,
                n_pixels_measured=int(n_after),
            )
        )

    return results
