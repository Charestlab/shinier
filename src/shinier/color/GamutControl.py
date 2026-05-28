"""
shinier.color.GamutControl
==========================

GamutControl implements robust gamut management for pipelines that manipulate luminance and chromaticity
separately (e.g., processing Y while preserving xy in CIE xyY). After such operations, some colors may become
out of gamut for the target RGB encoding: when converting back to linear RGB, one or more channels can fall
below 0 or exceed 1. Naïve clipping fixes the bounds but introduces non-linear, content-dependent distortions
(hue shifts, contrast changes) that can undermine the low-level statistics the pipeline is designed to control.

Strategies
----------
GamutControl provides five strategies (see `GAMUT_STRATEGY_TYPE`):

- clip:
  Rely on the converter's clipping/safe-mode behavior only (no global constraint estimation).

- constrain_image_luminance:
  For each image independently, compute a luminance scaling factor so that reconstructed linear RGB stays
  (mostly) within bounds, then scale Y accordingly (chromaticity is preserved).

- constrain_dataset_luminance:
  Compute a single global luminance scaling factor from the whole dataset and apply it to every image.
  This preserves relative luminance structure across images.

- constrain_image_chrominance:
  For each image independently, compute a per-image chroma scaling factor by desaturating xy toward the
  adapting white-point, keeping Y unchanged.

- constrain_dataset_chrominance:
  Compute one global chroma scaling factor for the dataset and apply it to all images by desaturating xy
  toward the adapting white-point.

Safeguards
----------
All non-trivial strategies include additional safeguards to make the estimated scaling factors robust:

- [S1] Low-luminance preventive desaturation (optional):
  In xyY, chromaticity becomes ill-conditioned near Y≈0: because x,y are normalized by (X+Y+Z), small numerical
  noise can produce unstable colours. Low-luminance pixels are therefore gently desaturated toward the white-point 
  chromaticity (Y unchanged).
    Implemented in: `apply_low_Y_desaturation`.

- [S2] Chroma validity mask (`chroma_ok`):
  Excludes ill-conditioned or physically implausible chromaticities (e.g., y near 0 or x+y slightly above 1)
  from driving the constraint estimation. These pixels may still be repaired later via clipping/safeguards.

- [S3] Reliability weighting by luminance:
  Pixels near Y≈0 and Y≈1 are down-weighted when estimating global factors because tiny xy noise at the
  luminance extremes can produce unrealistically large RGB excursions. Mid-tones therefore dominate the estimate.

- [S4] Allowed outlier clipping (`prc_clipping`):
  Optionally use a lower-quantile instead of the strict minimum when reducing per-pixel headroom ratios.
  This prevents a tiny fraction of extreme pixels from dictating aggressive global compression.

TODO : Quantify the impact of these strategies and safeguards on the final image statistics and on the order accuracy metrics.
    - i.e., the trade-off between gamut safety and distortion of the original statistics.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING, Literal, Optional
from pydantic import ConfigDict, PrivateAttr

from shinier.base import InformativeBaseModel
from shinier.color.Converter import ColorConverter
from shinier.utils import console_log, Bcolors
from shinier.Options import GAMUT_STRATEGY_TYPE
if TYPE_CHECKING:
    from shinier.ImageListIO import ImageListIO


class GamutControl(InformativeBaseModel):
    """Manage gamut constraints and repairs for image datasets.

    GamutControl is used when SHINIER modifies luminance while preserving
    chromatic information. It estimates whether reconstructed RGB values would
    fall outside the valid display gamut and either clips, compresses luminance,
    or desaturates chromaticity depending on ``strategy``.

    Attributes:
        color_space (Literal["xyY"]): Color space used for gamut repair.
        strategy (GAMUT_STRATEGY_TYPE): Gamut strategy to apply.
        rec_standard (str): RGB standard used by the internal color converters.
        warning_threshold (float): Scaling threshold below which warnings are logged.
        prc_clipping (float): Percentage of extreme pixels allowed to clip when
            robust quantile reduction is used.
        low_Y_desaturate (bool): Whether to desaturate low-luminance chroma before processing.
        low_Y_threshold (float): Low-luminance threshold in [0, 1].
        low_Y_fade_width (float): Width of the optional low-luminance fade ramp.
        log_low_Y_chroma_loss (bool): Whether to log chroma-loss metrics for low-luminance desaturation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Safeguard IDs used in comments below:
    # [S1] low-luminance preventive desaturation
    # [S2] chroma validity mask
    # [S3] reliability weighting by luminance
    # [S4] outlier clipping (quantile/min reduction)
    color_space: Literal['xyY'] = 'xyY'
    strategy: GAMUT_STRATEGY_TYPE = 'constrain_dataset_chrominance'
    rec_standard: str = 'rec709'
    warning_threshold: float = 1.0
    prc_clipping: float = 0.5  # (%) Let the most extreme outliers clip (must be less than 1%).

    low_Y_desaturate: bool = False
    low_Y_threshold: float = 0.01  # Y in [0,1]
    low_Y_fade_width: float = 0.0  # 0 -> hard threshold; >0 -> linear fade
    log_low_Y_chroma_loss: bool = False

    _converter: ColorConverter = PrivateAttr(default_factory=ColorConverter)
    _converter_raw: ColorConverter = PrivateAttr(default_factory=ColorConverter)

    def model_post_init(self, __context):
        self._converter = ColorConverter(rec_standard=self.rec_standard, safe_mode=True)
        self._converter_raw = ColorConverter(rec_standard=self.rec_standard, safe_mode=False)

    # ------------------------------------------------------------------
    # Small helpers (avoid repeating masks / reduction logic)
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_chroma_ok_xy(
            x: np.ndarray,
            y: np.ndarray,
            eps_y: float = 1e-3,
            eps_sum_xy: float = 1e-6) -> np.ndarray:
        """
        Return a boolean mask selecting numerically stable and physically plausible
        CIE xy chromaticities.

        Notes
        -----
        Theoretical chromaticity constraints are:

            x >= 0, y >= 0, and x + y <= 1

        where z = 1 - x - y must remain non-negative.

        Two small numerical safeguards are added:

        - y > eps_y:
            Chromaticity becomes ill-defined as y → 0 because the xyY→XYZ
            conversion contains divisions by y. Very small values lead to
            unstable tristimulus reconstruction and extreme RGB excursions.
            This threshold removes near-singular chromaticities.

        - x + y <= 1 - eps_sum_xy:
            Points exactly on the spectral boundary (x + y = 1) are numerically
            fragile. Floating-point rounding can otherwise cause sign flips in
            z = 1 - x - y and produce negative or unstable RGB values.
            The epsilon keeps chromaticities safely inside the valid simplex.

        These epsilons are therefore numerical stability margins rather than
        theoretical color constraints.
        """
        return (
            np.isfinite(x) & np.isfinite(y) &
            (y > eps_y) &
            (x >= 0.0) & (y >= 0.0) &
            (x + y <= 1.0 - eps_sum_xy)
        )

    @staticmethod
    def _reliability_from_Y01(Y01: np.ndarray, fade_width: float = 0.05) -> np.ndarray:
        """Reliability weight in [0,1], down-weighting near Y=0 and Y=1."""
        dist_to_edge = np.minimum(Y01, 1.0 - Y01)
        return np.clip(dist_to_edge / fade_width, 0.0, 1.0)

    @staticmethod
    def _min_or_quantile(values: np.ndarray, quantile_threshold: float | None) -> float:
        """Return min(values) or a lower-quantile if configured."""
        if values.size == 0:
            return 1.0
        if quantile_threshold is None:
            return float(np.min(values))
        return float(np.quantile(values, quantile_threshold))

    @staticmethod
    def _format_quantile_msg(quantile_threshold: float | None) -> str:
        """Format a short message describing quantile clipping."""
        if quantile_threshold is None:
            return ''
        return f'when {quantile_threshold * 100:.2f}% most extreme pixels clipped'

    def _log_image_overflow(
        self,
        kind: str,
        idx: int | None,
        local_min: float,
        quantile_threshold: float | None,
        verbose: bool,
    ) -> None:
        """Log a standardized per-image overflow message.

        Args:
            kind: Either 'luminance' or 'chrominance'.
            idx: Image index. If None, logs without an index label.
            local_min: Local minimum ratio (<=1 implies overflow).
            quantile_threshold: Quantile threshold used for reduction, if any.
            verbose: Whether to emit the log.
        """
        if (not verbose) or (local_min >= 1.0):
            return

        msg = self._format_quantile_msg(quantile_threshold)
        prefix = f'[GamutControl] Image {idx}: ' if idx is not None else '[GamutControl] Image-level '
        label = 'Luminance' if kind == 'luminance' else 'Chrominance'
        console_log(
            f"{prefix}{label} overflows by {(1 - local_min) * 100:3.2f}% {msg}",
            indent_level=1,
            verbose=verbose,
            color=Bcolors.WARNING,
        )

    # ------------------------------------------------------------------
    # Preventive low-luminance desaturation (xy only)
    # ------------------------------------------------------------------
    @staticmethod
    def _low_Y_strength(Y01: np.ndarray, threshold: float, fade_width: float) -> np.ndarray:
        """Strength in [0,1] for low-luminance desaturation.

        - strength=1 at Y=0
        - strength=0 at Y>=threshold
        - if fade_width>0: smoothstep ramp down between [threshold-fade_width, threshold]
        """
        if fade_width <= 0.0:
            return (Y01 < threshold).astype(np.float64)

        # Smoothstep ramp: 1 at (threshold - fade_width) and below, 0 at threshold and above.
        t = (threshold - Y01) / fade_width
        t = np.clip(t, 0.0, 1.0)

        # Smoothstep for a C1 transition (avoids a kink in the desaturation strength).
        # smoothstep(t) = 3t^2 - 2t^3
        return t * t * (3.0 - 2.0 * t)

    def _desaturate_xy_toward_white(self, xy: np.ndarray, strength: np.ndarray) -> np.ndarray:
        """Desaturate xy toward the converter white-point using per-pixel `strength` in [0,1]."""
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError(f"xy must be shape (H,W,2), got {xy.shape}")

        # Obtain white-point chromaticity in the xy color from XYZ coordinates
        W = self._converter.white_point
        W_sum = float(np.sum(W))
        xw, yw = float(W[0] / W_sum), float(W[1] / W_sum)

        s = np.asarray(strength, dtype=np.float64)
        if s.shape != xy.shape[:2]:
            raise ValueError(f"strength must be shape (H,W), got {s.shape} for xy {xy.shape}")
        
        # Linear chromatic desaturation toward white-point: xy_out = (1-s)*xy + s*white
        out = np.empty_like(xy, dtype=np.float64)
        out[..., 0] = (1.0 - s) * xy[..., 0] + s * xw
        out[..., 1] = (1.0 - s) * xy[..., 1] + s * yw
        return out

    def apply_low_Y_desaturation(
        self,
        Y: np.ndarray,
        other: np.ndarray,
        idx: int | None = None,
        verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """[S1] Optionally desaturate chroma for low-luminance pixels.

        Intended as a *preventive* step before luminance manipulation.
        Only modifies `other` (xy). Returns Y unchanged.

        Args:
            Y: Luminance image, either in [0,1] or [0,255]. Shape (H,W) or (H,W,1).
            other: Chromaticity (xy) of shape (H,W,2).
            idx: Optional image index for logging.
            verbose: Whether to emit informative logs.
        """
        if not self.low_Y_desaturate:
            return Y, other

        # Normalize Y to [0,1]
        Y_arr = np.asarray(Y)
        Y01 = Y_arr[..., 0] if (Y_arr.ndim == 3 and Y_arr.shape[-1] == 1) else Y_arr
        Y01 = Y01.astype(np.float64)
        if np.nanmax(Y01) > 1.5:
            Y01 = Y01 / 255.0

        # Calculate low-luminance desaturation strength.
        strength = self._low_Y_strength(Y01, threshold=self.low_Y_threshold, fade_width=self.low_Y_fade_width)
        if np.all(strength <= 0.0):
            return Y, other

        # Apply preventive desaturation toward white-point (Y unchanged).
        prc = float(np.mean(strength > 0.0))
        other_out = self._desaturate_xy_toward_white(other, strength=strength)

        msg = ''
        if verbose:
            prefix = f"[GamutControl] Image {idx}: " if idx is not None else "[GamutControl] "
            msg += f"{prefix}{prc*100:2.2f}% pixels desaturated for low luminance (Y < {self.low_Y_threshold})."

        # Optional chroma-loss metric (lazy import to avoid circular imports)
        if self.log_low_Y_chroma_loss and verbose:
            try:
                from shinier.color.quantify_chroma_loss import relative_mean_chroma_loss_pct_global_lab

                xy_before = np.asarray(other, dtype=np.float64)
                xy_after = np.asarray(other_out, dtype=np.float64)
                xyY_before = np.dstack([xy_before, Y01])
                xyY_after = np.dstack([xy_after, Y01])

                srgb_before = self._converter.xyY_to_sRGB(xyY_before)
                srgb_after = self._converter.xyY_to_sRGB(xyY_after)

                loss_pct, mean_c0, mean_c1 = relative_mean_chroma_loss_pct_global_lab(
                    converter=self._converter,
                    srgb_before_01=srgb_before,
                    srgb_after_01=srgb_after,
                )

                console_log(
                    msg=(
                        f"{msg} Mean chroma loss = {loss_pct:0.2f}%"
                        # f"(mean sqrt(aˆ2 + bˆ2): {mean_c0:0.3f} -> {mean_c1:0.3f})"
                    ),
                    indent_level=1,
                    verbose=verbose,
                    color=Bcolors.WARNING,
                )
            except (ImportError, AttributeError, ValueError):
                pass

        return Y, other_out

    def apply_dataset(self, buffer: ImageListIO, buffer_other: ImageListIO, verbose: bool = False) -> Tuple[ImageListIO, ImageListIO]:
        """Apply a *dataset-level* gamut strategy.

        Dataset-level strategies compute a single global factor (scale or desaturation) and apply it to all images.
        """
        if self.strategy == 'clip' or len(buffer) == 0:
            return buffer, buffer_other
        if self.strategy == 'constrain_dataset_luminance':
            return self._apply_constrain_dataset_luminance(buffer, buffer_other, verbose)
        if self.strategy == 'constrain_dataset_chrominance':
            return self._apply_constrain_dataset_chrominance(buffer, buffer_other, verbose)
        return buffer, buffer_other

    def apply_image(self, Y: np.ndarray, other: np.ndarray, idx: int = None, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Apply an *image-level* gamut strategy to a single image.

        Args:
            Y: Luminance image in [0,255] (float or uint) of shape (H,W).
            other: Chromaticity channels (e.g., xy) of shape (H,W,2).
            idx: Index of the image to be processed.
            verbose: Whether to emit informative logs.

        Returns:
            (Y_out, other_out) with same shapes as inputs.
        """
        if self.strategy == 'clip':
            return Y, other
        if self.strategy == 'constrain_image_luminance':
            return self._apply_constrain_image_luminance(Y, other, idx, verbose)
        if self.strategy == 'constrain_image_chrominance':
            return self._apply_constrain_image_chrominance(Y, other, idx, verbose)
        return Y, other

    def _apply_constrain_dataset_luminance(self, buffer: ImageListIO, buffer_other: ImageListIO, verbose: bool = False) -> Tuple[ImageListIO, ImageListIO]:
        """
        Global scaling along the luminance coordinate with Soft Singularity Protection.

        Prevents dark noise (e.g., Y=0.002) from triggering massive global dimming.
        """
        min_headroom = 1.0
        eps_y = 1e-3
        eps_sum_xy = 1e-6
        quantile_threshold = self.prc_clipping/100 if self.prc_clipping is not None else None
        for idx, Y in enumerate(buffer):
            Y_norm = Y / 255.0
            x, y = buffer_other[idx][..., 0], buffer_other[idx][..., 1]

            # --- 1. Compute raw luminance scaling to stay in gamut. ---
            # Compute the maximum in-gamut luminance for each chromaticity.
            # If Y exceeds this limit, the reconstructed RGB would clip.
            Y_max_map = self.get_max_luminance_map(x, y)
            with np.errstate(divide='ignore', invalid='ignore'):
                # How much we need to dim (e.g., 0.1 means dim by 90%).
                ratio_raw = Y_max_map / (Y_norm + 1e-9)
            
            # --- 2. [S3] Compute luminance reliability weights. ---
            # Mid-tones are trusted (w≈1); near-black/near-white pixels are down-weighted (w≈0).
            dist_to_edge = np.minimum(Y_norm, 1.0 - Y_norm)
            fade_width = 0.05
            w = np.clip(dist_to_edge / fade_width, 0.0, 1.0)

            # --- 3. [S2] Mask of the valid chromaticities (physically plausible and numerically stable). ---
            chroma_ok = self._mask_chroma_ok_xy(x, y, eps_y=eps_y, eps_sum_xy=eps_sum_xy)

            # --- 4. [S2][S3] Build a reliability-adjusted per-pixel scaling ratio. ---
            # Invalid chromaticities are forced to zero reliability.
            w = w * chroma_ok.astype(w.dtype)

            # Interpolate between the raw scaling ratio and 1.0 (no correction)
            # according to luminance/chroma reliability:
            #   w=1 -> full correction
            #   w=0 -> no correction
            #   0<w<1 -> partial correction
            ratio_safe = ratio_raw * w + 1.0 * (1.0 - w)

            # --- 5. Apply standard luminance masking for sanity (still useful for speed). ---
            mask = Y_norm > 0.001
            if np.any(mask):
                # --- 6. Create a single global scaling factor for this image. ---
                # Use either:
                #  - Minimum: satisfy the most constraining pixel (conservative but hugely affected by outliers)
                #  - [S4] Quantile: satisfy most pixels except a small fraction of outliers (more robust, less aggressive).
                if quantile_threshold is not None:
                    local_min = np.quantile(ratio_safe[mask], quantile_threshold)
                else:
                    local_min = np.min(ratio_safe[mask])

                self._log_image_overflow('luminance', idx, float(local_min), quantile_threshold, verbose)

                # --- 7. Keep the most restrictive image-level scaling ratio across the dataset. ---
                if local_min < min_headroom:
                    min_headroom = local_min

        scaling_factor = min(1.0, min_headroom)

        if scaling_factor < 1.0:
            self._log_warning('constrain_dataset_luminance', scaling_factor, "compressing", verbose=True)
            for idx in range(len(buffer)):
                buffer[idx] *= scaling_factor

        return buffer, buffer_other

    def _apply_constrain_dataset_chrominance(self, buffer: ImageListIO, buffer_other: ImageListIO, verbose: bool = False) -> Tuple[ImageListIO, ImageListIO]:
        """Dataset-level desaturation in xy with soft singularity protection.

        Computes one global chroma scaling factor (toward the achromatic white-point) that is safe for all images,
        while down-weighting unreliable chroma near luminance singularities (Y≈0 or Y≈1) and invalid xy pixels.

        Notes:
            - Works in xyY internally, but detects out-of-gamut in *raw* linear RGB (safe_mode=False).
            - Only modifies `buffer_other` (xy chromaticity). The luminance buffer is not modified here.
        """
        min_sat_ratio = 1.0
        quantile_threshold = self.prc_clipping / 100 if self.prc_clipping is not None else None
        eps_y = 1e-3
        eps_sum_xy = 1e-6
        eps_delta = 1e-9

        # --- 1. Build achromatic reference point in xy. ---
        # White-point toward which chromacities will be desaturated.
        W = self._converter.white_point
        W_sum = float(np.sum(W))
        xw, yw = float(W[0] / W_sum), float(W[1] / W_sum)

        for idx, Y in enumerate(buffer):
            Y_norm = Y / 255.0
            x, y = buffer_other[idx][..., 0], buffer_other[idx][..., 1]

            # --- 2. Compute raw chroma overflow evidence in linear RGB. ---
            # Values <0 or >1 indicate that the current chromaticity is out of gamut.
            xyz = self._converter_raw.xyY_to_xyz(np.dstack([x, y, Y_norm]))
            rgb = self._converter_raw.xyz_to_linRGB(xyz)

            # --- 3. Build achromatic gray reference at same luminance as the white point. ---
            # Chroma scaling is measured relative to the achromatic color with the same Y.
            gray_rgb = self._neutral_rgb_for_Y(Y_norm)

            # --- 4. [S3] Compute luminance reliability weights. ---
            w = self._reliability_from_Y01(Y_norm, fade_width=0.05)
        
            # --- 5. [S2] Mask of valid chromaticities. ---
            chroma_ok = self._mask_chroma_ok_xy(x, y, eps_y=eps_y, eps_sum_xy=eps_sum_xy)
        
            # --- 6. [S2][S3] Build a reliability-adjusted per-pixel chroma scaling candidate. ---
            # Invalid chromaticities are forced to zero reliability.
            w3 = (w * chroma_ok.astype(w.dtype))[..., None]  # (H,W,1)

            # --- 7. Compute per-channel safe chroma scaling factors. ---
            # delta is a 3-channel array storing the chromatic offset from neutral gray for each pixel.
            delta = rgb - gray_rgb

            # Desaturation needed to bring each channel back into gamut if it exceeds the bounds 
            #     k_high: Desaturation factor needed to bring channels above 1 back to 1.
            #     k_low: Desaturation factor needed to bring channels below 0 back to 0.
            with np.errstate(divide='ignore', invalid='ignore'):
                k_high = np.where(
                    (rgb > 1.0) & (delta > eps_delta),
                    (1.0 - gray_rgb) / delta,
                    1.0,
                )
                k_low = np.where(
                    (rgb < 0.0) & (delta < -eps_delta),
                    gray_rgb / (-delta),
                    1.0,
                )
            
            # --- 8. Build a safe per-pixel chroma scaling factor. ---
            # Minimum desaturation needed across channels to bring the pixel back in gamut.
            k_raw = np.minimum(k_high, k_low)
            # Reliability blend: w=0 -> k=1, w=1 -> k=k_raw
            k_safe = k_raw * w3 + (1.0 - w3)
            # Keep reductions well-defined and reduce to a single scalar per pixel by taking the minimum across channels.
            k_safe = np.where(np.isfinite(k_safe), k_safe, 1.0)
            k_map = np.min(k_safe, axis=-1)

            # --- 9. Apply standard masking before image-level reduction. ---
            mask = (Y_norm > 0.001) & chroma_ok & np.isfinite(k_map)
            
            # --- 10. Create a single global scaling factor for this image. ---
            # Use either:
            #  - Minimum: satisfy the most constraining pixel (conservative but hugely affected by outliers)
            #  - [S4] Quantile: satisfy most pixels except a small fraction of outliers (more robust, less aggressive).
            local_min_k = self._min_or_quantile(k_map[mask], quantile_threshold)

            # Per-image logging
            self._log_image_overflow('chrominance', idx, float(local_min_k), quantile_threshold, verbose)
            
            # --- 11. Keep the most restrictive image-level chroma scaling ratio across the dataset. ---
            if local_min_k < min_sat_ratio:
                min_sat_ratio = local_min_k

        scaling_factor = min(1.0, float(min_sat_ratio))

        if scaling_factor < 1.0:
            # Dataset-level warning
            self._log_warning('constrain_dataset_chrominance', scaling_factor, 'desaturating', verbose=True)

            # Apply global desaturation to all images in xy
            for idx in range(len(buffer_other)):
                x, y = buffer_other[idx][..., 0], buffer_other[idx][..., 1]
                x_new = xw + scaling_factor * (x - xw)
                y_new = yw + scaling_factor * (y - yw)
                buffer_other[idx] = np.dstack([x_new, y_new])

        return buffer, buffer_other
        
    def _apply_constrain_image_luminance(self, Y: np.ndarray, other: np.ndarray, idx: int = None, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Per-image scaling along Y with the same protections as the dataset method.

        Notes:
            This method only modifies Y (luminance). Chromaticity `other` is returned unchanged.
        """
        eps_y = 1e-3
        eps_sum_xy = 1e-6
        quantile_threshold = self.prc_clipping / 100 if self.prc_clipping is not None else None

        Y_norm = Y / 255.0
        if other.ndim != 3 or other.shape[-1] != 2:
            raise ValueError(f"other must be shape (H,W,2), got {other.shape}")
        x, y = other[..., 0], other[..., 1]

        # --- 1. Compute raw luminance scaling to stay in gamut. ---
        Y_max_map = self.get_max_luminance_map(x, y)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_raw = Y_max_map / (Y_norm + 1e-9)

        # --- 2. [S3] Compute luminance reliability weights. ---
        w = self._reliability_from_Y01(Y_norm, fade_width=0.05)

        # --- 3. [S2] Mask of valid chromaticities. ---
        chroma_ok = self._mask_chroma_ok_xy(x, y, eps_y=eps_y, eps_sum_xy=eps_sum_xy)

        # --- 4. [S2][S3] Build reliability-adjusted per-pixel scaling ratio. ---
        w = w * chroma_ok.astype(w.dtype)
        ratio_safe = ratio_raw * w + 1.0 * (1.0 - w)

        # --- 5. Apply standard masking + [S4] robust reduction for the image-level ratio. ---
        mask = (Y_norm > 0.001) & chroma_ok & np.isfinite(ratio_safe)
        local_min = self._min_or_quantile(ratio_safe[mask], quantile_threshold)
        self._log_image_overflow('luminance', idx, float(local_min), quantile_threshold, verbose)
        
        # --- 6. Apply final per-image scaling factor. ---
        scaling_factor = min(1.0, float(local_min))
        if scaling_factor >= 1.0:
            return Y, other

        Y_out = np.asarray(Y, dtype=np.float64) * scaling_factor
        return Y_out, other

    def _apply_constrain_image_chrominance(self, Y: np.ndarray, other: np.ndarray, idx: int = None, verbose: bool = False) -> Tuple[
        np.ndarray, np.ndarray]:
        """Per-image desaturation in xy by projecting toward the dataset white-point."""
        quantile_threshold = self.prc_clipping / 100 if self.prc_clipping is not None else None
        eps_y = 1e-3
        eps_sum_xy = 1e-6
        eps_delta = 1e-9

        Y_norm = Y / 255.0
        if other.ndim != 3 or other.shape[-1] != 2:
            raise ValueError(f"other must be shape (H,W,2), got {other.shape}")
        x, y = other[..., 0], other[..., 1]

        # --- 1. Build achromatic reference point in xy. ---
        W = self._converter.white_point
        W_sum = float(np.sum(W))
        xw, yw = float(W[0] / W_sum), float(W[1] / W_sum)

        # --- 2. Compute raw chroma overflow evidence in linear RGB. ---
        xyz = self._converter_raw.xyY_to_xyz(np.dstack([x, y, Y_norm]))
        rgb = self._converter_raw.xyz_to_linRGB(xyz)

        # --- 3. Build achromatic gray reference at same luminance as the white point. ---
        gray_rgb = self._neutral_rgb_for_Y(Y_norm)

        # --- 4. [S3] Compute luminance reliability weights. ---
        w = self._reliability_from_Y01(Y_norm, fade_width=0.05)

        # --- 5. [S2] Mask of valid chromaticities. ---
        chroma_ok = self._mask_chroma_ok_xy(x, y, eps_y=eps_y, eps_sum_xy=eps_sum_xy)

        # --- 6. [S2][S3] Build a reliability-adjusted per-pixel chroma scaling candidate. ---
        w3 = (w * chroma_ok.astype(w.dtype))[..., None]

        # --- 7. Compute per-channel safe chroma scaling factors. ---
        delta = rgb - gray_rgb
        with np.errstate(divide='ignore', invalid='ignore'):
            k_high = np.where(
                (rgb > 1.0) & (delta > eps_delta),
                (1.0 - gray_rgb) / delta,
                1.0,
            )
            k_low = np.where(
                (rgb < 0.0) & (delta < -eps_delta),
                gray_rgb / (-delta),
                1.0,
            )

        # --- 8. Build a safe per-pixel chroma scaling factor. ---
        k_raw = np.minimum(k_high, k_low)
        k_safe = k_raw * w3 + (1.0 - w3)
        k_safe = np.where(np.isfinite(k_safe), k_safe, 1.0)
        k_map = np.min(k_safe, axis=-1)

        # --- 9. Apply standard masking before image-level reduction. ---
        mask = (Y_norm > 0.001) & chroma_ok & np.isfinite(k_map)

        # --- 10. [S4] Create a single image-level chroma scaling factor. ---
        local_min_k = self._min_or_quantile(k_map[mask], quantile_threshold)

        self._log_image_overflow('chrominance', idx, float(local_min_k), quantile_threshold, verbose)

        # --- 11. Apply final per-image chroma scaling factor. ---
        scaling_factor = min(1.0, float(local_min_k))
        if scaling_factor >= 1.0:
            return Y, other

        x_new = xw + scaling_factor * (x - xw)
        y_new = yw + scaling_factor * (y - yw)
        other_out = np.dstack([x_new, y_new])
        return Y, other_out

    def _neutral_rgb_for_Y(self, Y01: np.ndarray) -> np.ndarray:
        """Return linear-RGB neutral gray corresponding to luminance Y.

        Builds an achromatic color defined by the current white-point
        chromaticity and the provided luminance values.

        Args:
            Y01: Luminance in [0,1], shape (H,W).

        Returns:
            Neutral linear RGB array of shape (H,W,3).
        """
        W = self._converter.white_point
        W_sum = float(np.sum(W))
        xw = float(W[0] / W_sum)
        yw = float(W[1] / W_sum)

        xyY_gray = np.dstack([
            np.full_like(Y01, xw, dtype=np.float64),
            np.full_like(Y01, yw, dtype=np.float64),
            Y01.astype(np.float64),
        ])

        xyz_gray = self._converter_raw.xyY_to_xyz(xyY_gray)
        rgb_gray = self._converter_raw.xyz_to_linRGB(xyz_gray)
        return rgb_gray
    
    def get_max_luminance_map(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the maximum luminance Y_max for each pixel that keeps the color defined by chromaticity (x, y).

        Arguments:
            x, y (ndarray): Chromaticity x and y coordinates.

        Returns:
            Y_max (ndarray): Maximum luminance before RGB clipping for the
            chromaticity (x, y) at each pixel. Same shape as the input arrays.
        """
        # Avoid division instability when converting xyY to XYZ
        y_safe = np.where(y <= 1e-9, 1e-9, y)

        # Reconstruct XYZ at unit luminance (Y = 1) for the given chromaticity
        X_test = x / y_safe
        Y_test = np.ones_like(x)
        Z_test = (1.0 - x - y) / y_safe
        xyz = np.stack([X_test, Y_test, Z_test], axis=-1)

        # Convert to linear RGB
        rgb_test = self._converter_raw.xyz_to_linRGB(xyz)

        # Find limiting RGB channel
        max_channel = np.max(rgb_test, axis=-1)

        # Compute luminance scaling
        Y_max = np.ones_like(x)
        mask = max_channel > 0
        Y_max[mask] = 1.0 / max_channel[mask]

        return np.clip(Y_max, 0.0, 1.0)

    def _log_warning(self, strategy: str, factor: float, action: str, verbose: bool = False):
        if factor < self.warning_threshold:
            console_log(f"[GamutControl] Strategy '{strategy}' is {action} dataset by {100*(1-factor):.1f}%.", indent_level=1, color=Bcolors.FAIL, verbose=verbose)
