# TODO: Before V1 commit: Remove all revision comments (e.g. see round in MatlabOperators)
# TODO: legacy_mode for SSIM optimization
# TODO: should target_hist, target_spectrum and target_sf be fixed for composite modes

# External package imports
from __future__ import annotations

import re
import warnings
from pathlib import Path
from unicodedata import is_normalized

import numpy as np
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view
from typing import (
    Any, Optional, Tuple, Union, NewType, List, Iterable,
    Callable, Literal, Dict, Annotated, TYPE_CHECKING, get_args, get_origin)
from PIL import Image
from itertools import chain

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local package imports
from . import _HAS_CYTHON
if TYPE_CHECKING:
    from .ImageProcessor import ImageProcessor
    from .ImageListIO import ImageListIO

if _HAS_CYTHON:
    from . import _cconvolve
from shinier import __version__ as package_version

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Bcolors:
    """
    Provides color-coding for terminal text output.
    """
    # --- ANSI color constants ---
    # COLOR_PROMPT = "\033[96m"  # bright cyan
    COLOR_TEXT = "\033[97m"  # bright white
    # COLOR_DEFAULT = "\033[93m"  # bright yellow
    # COLOR_INPUT = "\033[92m"  # bright green
    DEFAULT_TEXT = "\033[35m"
    CHOICE_VALUE = '\033[93m'
    GRAY = "\033[2m"
    MEDIUM_GRAY = "\033[38;5;247m"
    ALMOST_WHITE = "\033[38;5;255m"

    HEADER = '\033[95m'  # Processing steps
    OKBLUE = '\033[94m'  # Processing values
    OKCYAN = '\033[96m'  # Internal notes
    OKGREEN = '\033[92m'  # Ok values
    WARNING = '\033[93m'
    FAIL = '\033[91m'  # Problematic values
    ENDC = '\033[0m'  # Reset color
    BOLD = '\033[1m'  # Iteration
    UNDERLINE = '\033[4m'
    SECTION = '\033[4m\033[1m'  # Image loop
    SECTION_BRIGHT = '\033[4m\033[1m\033[97m'  # Image loop


def print_shinier_header(is_tty: bool = True, version: str = package_version):
    """Prints a styled header for the SHINIER CLI."""
    if is_tty:
        print("\033[2J")  # clear screen

    date_str = colorize(datetime.now().strftime("%Y-%m-%d %H:%M"), Bcolors.WARNING)

    banner = r"""
   ███████╗██╗  ██╗██╗███╗  ██╗██╗███████╗██████╗
   ██╔════╝██║  ██║██║████╗ ██║██║██╔════╝██╔══██╗
   ███████╗███████║██║██╔██╗██║██║█████╗  ██████╔╝
   ╚════██║██╔══██║██║██║╚████║██║██╔══╝  ██╔══██╗
   ███████║██║  ██║██║██║ ╚███║██║███████╗██║  ██║
   ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚══╝╚═╝╚══════╝╚═╝  ╚═╝
    """.strip("\n")

    console_log("")
    console_log(banner)
    console_log("")
    console_log(f"SHINIER — > Spectrum, Histogram, and Intensity Normalization, Equalization, and Refinement ({colorize(version, color=Bcolors.OKGREEN)})")
    console_log(f"Session started: {date_str}")
    console_log("─" * 60)
    console_log("")


class MatlabOperators:
    """
    Provides methods that replicate the behavior of MATLAB functions and operators
    in Python.

    This class is designed to provide static methods for MATLAB-like operations,
    aiming to mimic their behavior as closely as possible using NumPy and Python.
    It can be useful for porting MATLAB code to Python or when MATLAB-like behavior
    is desired in numerical computations.

    """

    @staticmethod
    def round(x):
        """
         alt_round(x)

         <x> is an array

         simulate the rounding behavior of matlab where 0.5 rounds
         to 1 and -.5 rounds to -1. (python rounds ties to the
         nearest even integer.)

         return:
          an array of rounded values

         example:
         import numpy as np
         x = np.array([-1, -0.5, 0, 0.5, 0.7, 1.0, 1.5, 2.1, 2.5, 2.6, 3.5])
         y = alt_round(x)

         from https://github.com/cvnlab/GLMsingle/blob/main/glmsingle/utils/alt_round.py

        return (np.sign(x) * np.ceil(np.floor(np.abs(x) * 2) / 2)).astype(int)
        ----
        Slight modifications to follow MATLAB's behavior of not changing types: Mathias Salvas-Hébert, 2025-07-16
         MATLAB :
            x = int8([-3.7, -1.2, 0.5, 2.9, 10.1]);
            y = round(x);
            class(y)
            > ans = 'int8'

            x = double([-3.7, -1.2, 0.5, 2.9, 10.1]);
            y = round(x);
            class(y)
            > ans = 'double'
        """
        return np.sign(x) * np.ceil(np.floor(np.abs(x) * 2) / 2)

    @staticmethod
    def uint8(x):
        """Replicates MATLAB's uint8 behavior: rounds and clips to [0, 255]"""
        return np.uint8(np.clip(MatlabOperators.round(x), 0, 255))

    @staticmethod
    def std2(x):
        """Replicates MATLAB's std2 function, which uses ddof=1"""
        return np.std(x, ddof=1)

    @staticmethod
    def mean2(x):
        return np.mean(x)

    @staticmethod
    def double(x):
        """Ensures double precision, similar to MATLAB's double."""
        return np.array(x, dtype=np.float64)

    @staticmethod
    def single(x):
        """Ensures single precision, similar to MATLAB's single."""
        return np.array(x, dtype=np.float32)

    @staticmethod
    def int16(x):
        """Replicates MATLAB's int16 behavior: rounds and clips to [-2**15, 2**15 - 1]"""
        return np.int16(np.clip(MatlabOperators.round(x), -2**15, 2**15 - 1))

    @staticmethod
    def int32(x):
        """Replicates MATLAB's int32 behavior: rounds and clips to [-2**31, 2**31 - 1]"""
        return np.int32(np.clip(MatlabOperators.round(x), -2**31, 2**31 - 1))

    @staticmethod
    def linspace(start, stop, num=50):
        """Replicates MATLAB's linspace function."""
        return np.linspace(start, stop, num)

    @staticmethod
    def ceil(x):
        """Replicates MATLAB's ceil function."""
        return np.ceil(x)

    @staticmethod
    def floor(x):
        """Replicates MATLAB's floor function."""
        return np.floor(x)

    @staticmethod
    def fix(x):
        """Replicates MATLAB's fix function (rounds toward zero)."""
        return np.sign(x) * np.floor(np.abs(x))

    @staticmethod
    def mod(x, y):
        """Replicates MATLAB's mod function (modulus operation with sign matching divisor)."""
        return np.mod(x, y)

    @staticmethod
    def rem(x, y):
        """Replicates MATLAB's rem function (remainder operation with sign matching dividend)."""
        return np.remainder(x, y)

    @staticmethod
    def rgb2gray(image):
        """Replicates MATLAB's rgb2gray function (ITU-R rec601)."""
        if image.ndim >= 3:
            from shinier.color.Converter import rgb2gray
            return rgb2gray(image=image[..., :3], conversion_type='rec601', matlab_601=True)
        else:
            return image


def get_field_values_from_pydantic_model(field):
    """Return all possible categorical values for a Pydantic field."""
    ann = field.annotation

    def extract_values(ann_type):
        """Recursively extract possible values from Literal/Union types."""
        origin = get_origin(ann_type)

        # Case 1: Literal[...] → return its args
        if origin is Literal:
            return list(get_args(ann_type))

        # Case 2: Union[...] → flatten all constituent possibilities
        if origin is Union:
            vals = []
            for arg in get_args(ann_type):
                if arg is type(None):
                    vals.append(None)  # explicitly include None
                else:
                    vals.extend(extract_values(arg))
            return vals

        # Case 3: bool field
        if ann_type is bool:
            return [True, False]

        # Case 4: Path → not categorical, but serialize default if defined
        if ann_type is Path:
            return [str(field.default)] if field.default is not None else [None]

        # Case 5: Non-categorical base types
        return []

    vals = extract_values(ann)
    default = field.default_factory() if getattr(field, "default_factory", None) is not None else field.default

    # Fallbacks: defaults or None if nothing categorical
    if not vals:
        vals = [default if default is not None else None]

    # Add default if not already present and meaningful
    if default is not None and default not in vals:
        vals.append(default)

    # Fallbacks: defaults or None if nothing categorical
    if not vals:
        if field.default is not None:
            return [field.default]
        return [None]

    # Add default if not already present and meaningful
    if field.default is not None and field.default not in vals:
        vals.append(field.default)

    # Deduplicate while preserving order
    seen = set()
    unique_vals = [v for v in vals if not (v in seen or seen.add(v))]
    return unique_vals


def generate_pydantic_key_value_dict(model_cls):
    """Return dict of field → possible values for a Pydantic model."""
    possible_values = {}
    default_values = {}
    for name, field in model_cls.model_fields.items():
        try:
            possible_values[name] = get_field_values_from_pydantic_model(field)
            default = field.default_factory() if getattr(field, "default_factory", None) is not None else field.default
            default_values[name] = str(default) if isinstance(default, Path) else default
        except Exception as e:
            possible_values[name] = [f"Error: {e!r}"]
            default_values[name] = None
    return possible_values, default_values


def imhist_plot(
    img: np.ndarray,
    bins: int = 256,
    figsize=(8, 6),
    dpi=100,
    title: Optional[str] = None,
    target_hist: Optional[np.ndarray] = None,
    binary_mask: Optional[np.ndarray] = None,
    descriptives: bool = False,
    ax: Optional[plt.Axes] = None,
    show_normalized_rmse: bool = False,
) -> Tuple[plt.Figure, Tuple[Any, Any, Any]]:

    """Displays an image with its histogram and optional descriptive statistics.

    The image is shown on top, with a compact horizontal histogram below.
    A grayscale gradient bar (0–255) is placed directly under the histogram.
    When `descriptives=True`, the histogram includes:
      * A vertical line indicating the mean (μ)
      * A translucent band spanning [μ − σ, μ + σ]
    For RGB images, μ and σ are computed and displayed per channel.

    Args:
        img (np.ndarray): Input image. Accepts (H, W) grayscale or (H, W, 3) RGB arrays.
            Alpha channels are ignored if present. Floating-point arrays are converted
            to uint8 for display (assuming [0, 1] range if max ≤ 1).
        bins (int, optional): Number of histogram bins in [0, 255]. Defaults to 256.
        figsize (tuple, optional): Matplotlib figure size. Defaults to (8, 6).
        dpi (int, optional): Matplotlib figure DPI. Defaults to 100.
        title (str | None, optional): Optional title for the image. Defaults to None.
        target_hist (np.ndarray | None, optional): If provided, overlays a
            target histogram. Defaults to None.
        binary_mask (np.ndarray | None, optional): Optional mask corresponding
            to image for computing histogram. Defaults to None.
        descriptives (bool, optional): If True, overlays mean (μ) and ±1σ on the
            histogram (per-channel for RGB). Defaults to False.
        ax (plt.Axes, optional): Axes on which to display the image. Defaults to None.
        show_normalized_rmse (bool, False): If True, shows normalized RMSE.

    Returns:
        tuple:
            fig (matplotlib.figure.Figure): The created matplotlib figure.
            (ax_img, ax_bar, ax_hist): Tuple of matplotlib.axes.Axes for the image,
            gradient bar, and histogram, respectively.
    """

    # --- Make sure the image is in the [0, 255] range ---
    if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
        img *= 255

    if img.ndim == 3:
        if np.all(img[..., :3] == img[..., :1]):
            # If grayscale image with 3 channels, remove redundancy
            img = img[..., 0]

    # --- normalize input image to uint8; drop alpha if present ---
    arr = im3D(img)
    is_rgb = arr.shape[2] == 3
    arr = arr[..., :min(3, arr.shape[2])]
    if target_hist is not None and target_hist.shape[1] != arr.shape[2]:
        raise ValueError("target_hist 2nd dimension should match image's third dimension.")

    # --- histograms for displayed image ---
    hist_normalized = imhist(image=arr, mask=binary_mask, n_bins=256, normalized=True)
    Hmax = hist_normalized.max()
    edges = np.linspace(0, 256, bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # --- figure & axes ---
    fontname = 'Arial'
    ax_img = None
    ax_hist = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.5, 1.4], hspace=0.12)
        ax_img = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])
        _ = imshow(arr, ax=ax_img)
        if title:
            ax_img.set_title(title, fontsize=11, fontname=fontname)

        # Align histogram width to image width
        fig.canvas.draw()
        img_pos = ax_img.get_position()
    else:
        fig = ax.figure
        ax_hist = ax

    hist_pos = ax_hist.get_position()
    if ax is None:
        ax_hist.set_position([img_pos.x0, hist_pos.y0, img_pos.width, hist_pos.height])

    # --- plot histogram(s) of displayed image ---
    colors = ['red', 'green', 'blue'] if is_rgb else ['black']
    labels = ['R', 'G', 'B'] if is_rgb else ['Image']
    for ch in range(arr.shape[2]):
        ax_hist.plot(centers, hist_normalized[:, ch], color=colors[ch], lw=1, label=labels[ch])
        # ----------------- optional TARGET histogram overlay -----------------
        if target_hist is not None:
            # expects your helper to return target histogram aligned with `centers`
            # target_hist, initial_hist = _hist_match_target(target_images, target_masks, normalized=normalize)
            ax_hist.plot(centers, target_hist[:, ch], ls='--', lw=1, color=colors[ch], label=f'Target {labels[ch]}')

    # --- Normalized RMSE computation and display ---
    if show_normalized_rmse:
        if target_hist is not None:
            nrmse = normalized_rmse(hist_normalized, target_hist, mode="assume [0, 1]")
            rmse_text = "NRMSE = {:1.2e}".format(nrmse)
            ax_hist.text(0.05, 0.98, rmse_text, transform=ax_hist.transAxes, ha="left", va="top", fontsize=9, fontname=fontname)
    ax_hist.set_xlim(0, 255)
    ax_hist.set_ylim(0, Hmax * 1.05 if Hmax > 0 else 1)
    ax_hist.set_yticks([])
    ax_hist.set_xticks([])  # no ticks on the histogram axis
    ax_hist.set_ylabel("Frequency", fontname=fontname, labelpad=8)
    for spine in ("top", "right"):
        ax_hist.spines[spine].set_visible(False)

    # --- grayscale gradient bar directly under the histogram axis ---
    divider = make_axes_locatable(ax_hist)
    ax_bar = divider.append_axes("bottom", size="4%", pad=0.0)  # pad=0.0 to stick to the axis
    gradient = np.linspace(0, 1, 256, dtype=np.float64).reshape(1, -1)
    ax_bar.imshow(gradient, cmap='gray', aspect='auto', extent=[0, 255, 0, 1])
    ax_bar.set_xlim(ax_hist.get_xlim())
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    xlabel_text = "Pixel intensity"
    ax_bar.set_xlabel(xlabel_text, fontname=fontname, labelpad=2)

    # ----------------- descriptives overlay (μ and ±1σ) -----------------
    if descriptives:
        with plt.rc_context({"font.family": fontname}):
            y_top = ax_hist.get_ylim()[1]
            alpha_band = 0.15
            text_kwargs = dict(fontsize=9, ha='left', va='top')

            if is_rgb:
                # Per-channel stats
                stats = [
                    ("R", arr[..., 0].ravel(), "red"),
                    ("G", arr[..., 1].ravel(), "green"),
                    ("B", arr[..., 2].ravel(), "blue"),
                ]
                # stagger text vertically
                y_texts = np.linspace(y_top * 0.98, y_top * 0.86, num=3)
                for (label, data, c), y_txt in zip(stats, y_texts):
                    mu = float(np.mean(data))
                    sd = float(np.std(data, ddof=0))
                    # band
                    ax_hist.axvspan(mu - sd, mu + sd, color=c, alpha=alpha_band, lw=0)
                    # mean line
                    ax_hist.axvline(mu, color=c, lw=1.8)
                    # text
                    ax_hist.text(mu + 3, y_txt, f"{label}: μ={mu:.1f}, σ={sd:.1f}", color=c, **text_kwargs)
            else:
                data = arr.ravel()
                mu = float(np.mean(data))
                sd = float(np.std(data, ddof=0))
                ax_hist.axvspan(mu - sd, mu + sd, color='black', alpha=alpha_band, lw=0)
                ax_hist.axvline(mu, color='black', lw=1.8)
                ax_hist.text(mu + 3, y_top * 0.95, f"μ={mu:.1f}, σ={sd:.1f}", color='black', **text_kwargs)

    # legend if anything was added (set font explicitly)
    handles, labels = ax_hist.get_legend_handles_labels()
    if handles:
        leg = ax_hist.legend(frameon=False, fontsize=9, loc='upper right')
        for text in leg.get_texts():
            text.set_fontname(fontname)

    # Ensure layout is updated
    if ax is None:
        fig.show()

    # Enforce font for any ticks that might be enabled later
    for label in ax_hist.get_xticklabels() + ax_hist.get_yticklabels():
        label.set_fontname(fontname)

    return fig, (ax_img, ax_bar, ax_hist)


def freq_axis(n: int) -> np.ndarray:
    """Compute spatial frequency axis for image spectrum"""
    # MATLAB:
    # even n:   -n/2 : n/2-1
    # odd  n:   -n/2 : n/2-1  (with halves → -2.5,-1.5,...,+1.5 for n=5)
    if n % 2 == 0:
        return np.arange(-n//2, n//2, dtype=np.float64)
    else:
        # center at half-steps to avoid 0; e.g., n=5 -> -2.5..+1.5
        half = n // 2
        return np.arange(-(half + 0.5), half + 0.5, 1.0, dtype=np.float64)


def get_radius_grid(x_size: int, y_size: int, legacy_mode: bool = False) -> np.ndarray:
    """Compute the radius grid for rotational average"""
    f2 = freq_axis(x_size)  # rows
    f1 = freq_axis(y_size)  # cols
    XX, YY = np.meshgrid(f1, f2)  # shape (xs, ys)

    # --- polar radius, MATLAB rounding rule ---
    r = np.hypot(XX, YY)
    r_adjustment = -1 if (x_size % 2 == 1) or (y_size % 2 == 1) else 0
    r = MatlabOperators.round(r) if legacy_mode else np.round(r, decimals=0) + r_adjustment

    # Non-negative integer bin indices
    return np.clip(r, 0, None).astype(np.int64)


def rotational_avg(spectrum: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Mean spectrum per radius bin (annular average)."""
    # Precompute counts per radius (for true rotational *averages*, not sums)
    r = radius if radius.ndim == 1 else radius.ravel()
    weights = spectrum if spectrum.ndim == 1 else spectrum.ravel()
    counts = np.bincount(r)
    counts[counts == 0] = 1  # protect against divide-by-zero

    sums = np.bincount(r, weights=weights)
    return sums / counts


def imshow(image: np.ndarray, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Show an image with matplotlib axes.
    Args:
        image: An image
        ax: Optional[plt.Axes], default = None. An axe to display the image on.

    Returns:

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    image = im3D(image)
    if image.shape[2] == 1:
        from shinier.color.Converter import gray2rgb
        image = gray2rgb(image)

    # If float with [0, 255] range, convert to [0, 1]
    if np.issubdtype(image.dtype, np.floating) and image.max() > 2:
        image /= 255

    ax.imshow(image)
    ax.axis('off')

    return fig, ax


def sf_profile(
        image: np.ndarray,
        spectrum: Optional[np.ndarray] = None,
        is_power_spectrum: bool = True,
        is_truncated: bool = True,
        legacy_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotational average of the Fourier (energy) spectrum.

    Args:
        image: np.ndarray
            Input image.
        spectrum: np.ndarray, default = None
            If not None, uses spectrum instead of computing new spectrum on input image.
        is_power_spectrum: bool, default = True
            If True, computes power spectrum else uses magnitude.
        is_truncated: bool, default = True
            If True, truncates sf profile to the nyquist of the shortest image size.
        legacy_mode: bool, default = False
            If True, uses Matlab round function
    Returns: Tuples[np.ndarray, np.ndarray]
        Rotational average of spectrum
        radius
    """

    # --- frequency grids replicating MATLAB logic ---

    # --- Fourier energy (fft-shifted) ---
    image = im3D(image)
    exp = 2 if is_power_spectrum else 1
    if spectrum is None:
        magnitude, _ = image_spectrum(image)
        power_spectrum = magnitude ** exp
    else:
        power_spectrum = spectrum ** exp

    # Compute radius
    xs, ys, channels = image.shape
    r = get_radius_grid(x_size=xs, y_size=ys, legacy_mode=legacy_mode)

    # --- accumarray equivalent: mean energy per radius ---
    rot_avg = []
    radians = []
    for ch in range(channels):
        avg = rotational_avg(spectrum=power_spectrum[..., ch], radius=r)
        radii = np.arange(1, avg.shape[0] + 1)

        # Match MATLAB: avg = avg(2:floor(min(xs,ys)/2)+1)
        if is_truncated:
            R = int(np.floor(min(xs, ys) / 2.0))
            radii = np.arange(1, R + 1)
            avg = avg[1:R + 1]
        rot_avg.append(avg)
        radians.append(radii)
    rot_avg = np.array(rot_avg).T
    radians = np.array(radians).T
    return rot_avg, radians


def sf_plot(
        image: np.ndarray,
        sf_p: Optional[np.ndarray] = None,
        target_sf: Optional[np.ndarray] = None,
        ax: Optional[plt.axis] = None,
        show_normalized_rmse: bool = False,
) -> Union[plt.Figure, plt.Axes]:
    """
    Rotational average of the Fourier energy spectrum.

    Args:
        image : np.ndarray
            Image array of shape (H, W) or (H, W, 3). Can be uint8 or float.
        sf_p : np.ndarray, default None
            If not None, uses sf_p (spatial frequency profile) instead of generating a new spectrum.
        target_sf : np.ndarray, default None
            If not None, display target_sf against sf_p.
        ax : plt.Axes, default None
            If not None, uses the ax instead of generating a new figure.
        show_normalized_rmse: bool, default = False
            If True, show normalized RMSE on graph.

    Returns:
        fig, ax : plt.Figure, plt.Axes
    """

    image = im3D(image)
    xs, ys, channels = image.shape
    R = int(np.floor(min(xs, ys) / 2.0))
    is_rgb = image.shape[2] == 3
    if target_sf is not None and target_sf.shape[0] > xs/2:
        target_sf = target_sf[1: R + 1]

    rot_avg, radii = sf_profile(image)
    rot_avg = rot_avg if sf_p is None else sf_p

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.figure

    colors = ['red', 'green', 'blue'] if is_rgb else ['black']
    labels = ['R', 'G', 'B'] if is_rgb else ['Image']
    for ch in range(channels):
        ax.loglog(radii[:, ch], rot_avg[:, ch], color=colors[ch], label=labels[ch])
        # ----------------- optional TARGET histogram overlay -----------------
        if target_sf is not None:
            ax.loglog(radii[:, ch], target_sf[:, ch], ls='--', lw=1, color=colors[ch], label=f'Target {labels[ch]}')

    # --- Normalized RMSE computation and display ---
    if show_normalized_rmse:
        if target_sf is not None:
            nrmse = normalized_rmse(target_sf, rot_avg, mode="actual range")
            rmse_text = "NRMSE = {:1.2e}".format(nrmse)
            ax.text(0.5, 0.98, rmse_text, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    ax.legend(frameon=False)
    ax.set_xlabel('Spatial frequency (cycles/image)')
    ax.set_ylabel('Energy')

    return fig


def spectrum_plot(
        spectrum: np.ndarray,
        cmap: str = "gray",
        log: bool = True,
        gamma: float = 1.0,
        ax: Optional[plt.Axes] = None,
        with_colorbar: bool = True,
        colorbar_label: str = 'log(1 + |F|) (stretched)',
        target_spectrum: Optional[np.ndarray] = None,
        show_normalized_rmse: bool = False,
    ) -> Union[plt.Figure, plt.Axes]:

    """Display a Fourier magnitude spectrum with optional log and gamma scaling."""
    if plt is None:
        raise RuntimeError(
            "Matplotlib is not installed. "
            "Install with: pip install shinier[dev]"
        )

    spec = np.abs(spectrum).astype(np.float64)

    # log scaling
    if log:
        spec = np.log1p(spec)

    # stretch to [0,1]
    spec = stretch(spec)

    # gamma correction
    if gamma != 1.0:
        spec = spec ** gamma

    # Axis in cycles/image (cpi) : d=1/N
    xs, ys = spec.shape[:2]  # rows, cols
    f_x = np.fft.fftshift(np.fft.fftfreq(ys, d=1 / ys))
    f_y = np.fft.fftshift(np.fft.fftfreq(xs, d=1 / xs))

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    else:
        fig = ax.figure

    implot = ax.imshow(
        spec, cmap=cmap,
        extent=(f_x.min(), f_x.max(), f_y.min(), f_y.max())
    )

    if with_colorbar:
        if ax is not None:
            fig = ax.figure
        fig.colorbar(implot, ax=ax, label=colorbar_label)

    if show_normalized_rmse:
        if target_spectrum is not None:
            nrmse = normalized_rmse(stretch(target_spectrum), stretch(spectrum), mode="assume [0, 1]")
            rmse_text = "NRMSE = {:1.2e}".format(nrmse)
            ax.text(0.5, 1.02, rmse_text, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Spatial frequency (cycles/image)")
    ax.set_ylabel("Spatial frequency (cycles/image)")
    # fig.tight_layout()

    return fig


def im_power_spectrum_plot(im: np.ndarray, with_colorbar: bool = True):
    """2D log-scaled Fourier power spectrum (centered).

    Visualizes the distribution of image energy across spatial frequencies and orientations.
    The center of the plot corresponds to low spatial frequencies, while the edges represent
    high frequencies. The brightness at each point indicates the amplitude |F(u, v)| — that
    is, the energy contribution of a given spatial frequency (radial distance) and
    orientation (angle).

    Args:
        im : np.ndarray
            Image array of shape (H, W) or (H, W, 3). Can be uint8 or float.
            RGB is converted to luminance (ITU-R BT.601).
        with_colorbar : bool, default True
            If True, show the colorbar on the right side.

    Returns:
        fig : Matplotlib image
    """
    # --- to grayscale float64 ---
    arr = np.asarray(im)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        from shinier.color.Converter import rgb2gray
        # suppose rgb2gray dispo; sinon fais la combinaison manuelle
        gray = rgb2gray(arr, conversion_type='rec709').astype(np.float64, copy=False)
    else:
        gray = arr.astype(np.float64, copy=False)

    # Power spectrum (centered)
    magnitude, _ = image_spectrum(gray)
    spec = magnitude ** 2
    fig = spectrum_plot(spectrum=spec, cmap='gray', with_colorbar=with_colorbar, colorbar_label='log(1 + |F|²) (normalized)')
    return fig


def stretch(arr: np.ndarray) -> np.ndarray:
    """Stretch an array to the range [0, 1].

    This rescales the input array so that its minimum maps to 0 and its
    maximum maps to 1. Works for any number of dimensions (grayscale,
    color images, or higher-dimensional data).

    Args:
        arr (np.ndarray): Input array of any shape and numeric dtype.

    Returns:
        np.ndarray: Array of the same shape as input, dtype float64,
        with values scaled to [0, 1].

    Notes:
        - If the array has constant values (max == min), returns an array
          of zeros (to avoid division by zero).
        - Output is float64 for numerical stability. Cast to float32 if
          needed for memory/performance reasons.
    """
    arr = np.asarray(arr, dtype=np.float64)
    min_val = arr.min()
    max_val = arr.max()

    if np.isclose(max_val, min_val):
        return np.zeros_like(arr, dtype=np.float64)

    return (arr - min_val) / (max_val - min_val)


def convolve_1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Apply 1D convolution with reflect padding along a chosen axis.

    Args:
        arr (np.ndarray): Input 2D image array.
        kernel (np.ndarray): 1D convolution kernel.
        axis (int): Axis along which to convolve (0 for vertical, 1 for horizontal).

    Returns:
        np.ndarray: Convolved image.
    """
    if arr.ndim != 2:
        raise TypeError("Input must be a 2D array.")
    if kernel.ndim != 1:
        raise TypeError("Kernel must be 1D.")

    r = len(kernel) // 2
    if axis == 0:
        pad = ((r, r), (0, 0))
    elif axis == 1:
        pad = ((0, 0), (r, r))
    else:
        raise ValueError("axis must be 0 or 1")
    padded = np.pad(arr, pad, mode="reflect").astype(np.float64, copy=False)
    windows = sliding_window_view(padded, window_shape=len(kernel), axis=axis)

    # True convolution: reverse kernel (flip is redundant if kernel is symmetric)
    return np.tensordot(windows, kernel[::-1], axes=([-1], [0]))


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 2D image with a kernel.

    Supports:
      * 1D kernel: applies separable convolution (horizontal then vertical).
      * 2D square kernel: applies dense 2D convolution with sliding windows.

    Args:
        image (np.ndarray): Input 2D image.
        kernel (np.ndarray): Convolution kernel. Either 1D (length k) or 2D (k x k).

    Returns:
        np.ndarray: Convolved image.

    Raises:
        TypeError: If inputs have wrong type or dimensionality.
        ValueError: If a 2D kernel is not square.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a np.ndarray")
    if image.ndim != 2:
        raise TypeError("Image must be 2D")
    if not isinstance(kernel, np.ndarray):
        raise TypeError("Kernel must be a np.ndarray")

    # Use compiled versions if exist
    if _HAS_CYTHON:
        if kernel.ndim == 1:
            return _cconvolve.convolve2d_separable(
                image.astype(np.float64, copy=False),
                kernel.astype(np.float64, copy=False)
            )
        elif kernel.ndim == 2:
            return _cconvolve.convolve2d_direct(
                image.astype(np.float64, copy=False),
                kernel.astype(np.float64, copy=False)
            )

    # Separable path: user passed a 1D kernel (e.g., Gaussian vector)
    if kernel.ndim == 1:
        tmp = convolve_1d(image, kernel, axis=1)
        return convolve_1d(tmp, kernel, axis=0)

    # Dense 2D path
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("2D kernel must be square")
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(image, pad, mode="reflect")
    windows = sliding_window_view(padded, (k, k))
    return np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))


def has_duplicates(image: np.ndarray, binary_mask: np.ndarray) -> bool:
    """
    Determines whether the given image contains duplicate pixel values on each channel.

    Args:
        image (np.ndarray): A numpy array representing the image data, where
            each element corresponds to a pixel value.
        binary_mask (np.ndarray): A numpy array of bools representing masked regions of the image.

    Returns:
        bool: True if duplicate pixel values are found in one channel of the image. False otherwise.
    """
    im = im3D(image)
    binary_mask = im3D(binary_mask)
    return any(n_unique(im[..., c][binary_mask[..., c]]) != binary_mask[..., c].sum() for c in range(im.shape[2]))


def n_unique(arr: np.ndarray) -> int:
    """Compute the number of unique values in 2D and 3D images efficiently using hash.

    Handles 1D vectors or 2D and 3D images. If the array is 3D, it is reshaped to
    (H*W, C). If it is 2D and square, it is treated as a single-channel
    image (converted via im3D). Otherwise, the shape is preserved.

    The implementation uses the np.view(void) hashing trick to perform
    fast and deterministic uniqueness checks across all vector dimensions.

    Args:
        arr (np.ndarray): Input array representing feature responses or image data.
            Can be:
                - (N) → single vector
                - (H, W) → single-channel image
                - (H, W, C) → multi-channels image
                - (N, D) → generic feature vectors

    Returns:
        int: Number of unique row vectors in the flattened representation.

    Raises:
        ValueError: If the input has fewer than 2 dimensions or empty shape.

    Notes:
        - The function avoids deep copies; all reshaping uses views where possible.
        - Deterministic (lexicographically consistent) results.
        - Faster (~3×) than np.unique(..., axis=0) for large 2D or 3D arrays.

    """

    # --- Handle 1D case ---
    if arr.ndim == 1:
        # Fast path: 1D scalar values
        return np.unique(arr).size

    # --- Handle 2D case ---
    if arr.ndim == 2:
        H, W = arr.shape
        if H == W:  # square → treat as image with 1 channel
            arr = arr[..., None]
        else:
            # Already vector data (e.g. N×D)
            b = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
            return np.unique(b).size

    # --- Now guaranteed 3D ---
    H, W, C = arr.shape
    FR_flat = arr.reshape(H * W, C)

    # --- Compute unique row hashes (fast & deterministic) ---
    b = FR_flat.view(np.dtype((np.void, FR_flat.dtype.itemsize * FR_flat.shape[1])))
    n_u = np.unique(b).size

    return int(n_u)


def strict_ordering(
        image: np.ndarray,
        kernels: list[np.ndarray],
        early_stop: bool = False,
        min_kernels: Optional[int] = None) -> Tuple[np.ndarray, List[float]]:
    """Assign strict pixel ordering using a customizable set of kernels.

    For each channel, this function applies a series of convolution kernels
    and stacks the resulting responses into a multidimensional feature vector
    used for lexicographic sorting. The order accuracy (OA) is tracked after
    each kernel addition.

    Args:
        image (np.ndarray): Input grayscale or color image (H, W[, C]).
        kernels (list[np.ndarray]): List of convolution kernels to apply.
        early_stop (bool): If True, the function will stop early once all pixels
            have unique feature responses (OA = 1.0) after applying at least
            `min_kernels` kernels.
        min_kernels (Optional[int]): Minimum number of kernels to apply before
            early stopping is allowed. Defaults to len(kernels) if
            early_stop=False, or ceil(len(kernels)/2) if early_stop=True.

    Returns:
        Tuple[np.ndarray, List[float]]:
            - im_sort: (H, W, C) array of lexicographic rank indices.
            - OA: List of order accuracies (float) for each channel.
    """
    im = im3D(image)
    M, N, P = im.shape

    # Normalize kernels (sum to 1 when possible)
    K = [(k / np.sum(k)) if np.sum(k) != 0 else k for k in kernels]
    nK = len(K)
    if min_kernels is None:
        min_kernels = int(np.ceil(nK / 2)) if early_stop else nK

    # One extra slot for the identity (raw channel)
    feat_dims = nK + 1
    im_sort = []
    OA = []
    for c in range(P):
        ch = im[:, :, c].astype(np.float64, copy=False)
        FR = np.zeros((M, N, feat_dims), dtype=np.float64)
        FR[:, :, 0] = ch

        oa, FR_flat = 0.0, None
        for idx, kernel in enumerate(K):
            FR[..., idx + 1] = convolve_2d(ch, kernel)

            # Optional compute for early stop and mandatory compute on last iteration
            if (idx + 1 >= min_kernels and early_stop) or (idx == nK - 1):
                used_dims = idx + 2  # identity + kernels up to idx
                FR_flat = FR[..., :used_dims].reshape(M * N, used_dims)
                n_u = n_unique(FR_flat)
                oa = n_u / (M * N)
                if oa == 1.0:
                    break

        # Lexicographic ordering
        idx_pos = np.lexsort(FR_flat[:, ::-1].T)
        idx_rank = np.argsort(idx_pos).reshape(M, N)

        OA.append(oa)
        im_sort.append(idx_rank)

    return np.stack(im_sort, axis=-1), OA


def exact_histogram(
    image: np.ndarray,
    target_hist: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    n_bins: Optional[int] = None,
    tie_strategy: Literal['none', 'moving-average', 'gaussian', 'noise', 'hybrid'] = "hybrid",
    verbose: bool = True
) -> Tuple[np.ndarray, List]:
    """Unified exact histogram specification.

    Provides a single entry point for 3 families of exact histogram specification strategies:
        1) Direct mapping assuming no isoluminant pixel (no ties): 'none'
        2) Coltuc's ordering based on filter bank: 'moving-average' or 'gaussian'.
        3) Noise-based tie-breaking: 'noise'.
        4) A hybrid strategies first using 'gaussian', then 'noise if ties persist: 'hybrid'.

    Args:
        image (np.ndarray): Input grayscale or RGB image.
        target_hist (np.ndarray): Target histogram counts/weights, shape (n_bins, C).
        binary_mask (Optional[np.ndarray]): Foreground mask.
        n_bins (Optional[int]): Number of bins; required for float images.
        tie_strategy (str): Strategy for tie-breaking. One of:
            "none", "moving-average", "gaussian", "noise", "hybrid" (default).
        verbose (bool): If True, logs key operations.

    Returns:
        Tuple[np.ndarray, List]:
            - im_out: Histogram-specified image.
            - OA: Order accuracy list per channel.
    """
    # --- Validate and prepare inputs ---
    L = n_bins if n_bins is not None else None
    if L is None and np.issubdtype(image.dtype, np.integer):
        L = 2 ** np.iinfo(image.dtype).bits

    if L is None:
        raise ValueError("L, the expected number of values per channel, must be specified!")

    image = im3D(image)
    x_size, y_size, n_channels = image.shape

    if tie_strategy not in ['none', 'moving-average', 'gaussian', 'noise', 'hybrid']:
        raise ValueError("tie_strategy must be one of ['none', 'moving-average', 'gaussian', 'noise', 'hybrid']")
    if image.dtype not in (np.uint8, np.uint16) and not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Input image must be 8- or 16-bit or float")
    if len(target_hist) != L:
        raise ValueError("Number of histogram bins must match maximum gray levels.")
    if target_hist.ndim != 2 or target_hist.shape[1] != n_channels:
        raise ValueError("Target histogram shape mismatch.")
    if binary_mask is not None:
        binary_mask = im3D(binary_mask)
        if binary_mask.shape != image.shape:
            raise ValueError("binary_mask must have same shape as image.")
        if np.sum(binary_mask) < (50 * n_channels):
            raise ValueError("Too few foreground pixels in binary mask.")
    else:
        binary_mask = np.ones(image.shape, dtype=bool)
    if n_channels not in [1, 3]:
        raise ValueError("Input image must have 1 or 3 channels.")

    # --- Prepare filters ---
    if tie_strategy == 'moving-average':
        # Coltuc kernels
        F2 = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.float64)
        g3 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        F4 = np.ones((5, 5), dtype=np.float64)
        F4[[0, 0, 1, 1, 1, 3, 3, 4, 4, 4],
           [0, 1, 0, 1, 4, 0, 4, 1, 3, 4]] = 0.0
        F5 = np.ones((5, 5), dtype=np.float64)
        F5[[0, 0, 4, 4], [0, 4, 0, 4]] = 0.0
        g6 = np.ones(5, dtype=np.float64)
        kernels = [F2, g3, F4, F5, g6]
        early_stop = False
        min_kernels = 5
    elif tie_strategy in ['gaussian', 'hybrid']:
        kernels = [
            gaussian_kernel(size, coverage=0.95, n_dim=1)
            for size in [3, 5, 7, 9, 17, 33, 65]
        ]
        early_stop = True
        min_kernels = 5

    # --- Tie-breaking strategy selection ---
    OA = 1
    if tie_strategy in ("moving-average", "gaussian", "hybrid"):
        im_sort, OA = strict_ordering(image, kernels, early_stop=early_stop, min_kernels=min_kernels)
        OA = np.array(OA)

        # Optional noise fallback
        if tie_strategy == "hybrid" and np.any(OA < 1.0):
            if verbose:
                n_iso = np.sum((1 - OA) * x_size * y_size)
                msg = f"[exact_histogram] {n_iso:1.0f} isoluminant pixels detected → applying noise (0.1)."
                console_log(msg=msg, indent_level=1, color=Bcolors.WARNING, verbose=True)
            im_sort = im_sort.astype(np.float64, copy=True)
            im_sort += np.random.uniform(-0.1, 0.1, size=image.shape)

    if tie_strategy in ["noise", "hybrid"]:
        hybrid_extra_step = tie_strategy == 'hybrid' and np.any(OA < 1.0)
        if hybrid_extra_step:
            if verbose:
                n_iso = np.sum((1 - OA) * x_size * y_size)
                msg = f"[exact_histogram] {n_iso:1.0f} isoluminant pixels detected → applying noise (0.1)."
                console_log(msg=msg, indent_level=1, color=Bcolors.WARNING, verbose=True)
            noise_level = 0.1
        elif tie_strategy == 'noise':
            noise_level = tie_breaking_noise_level(image)
            im_sort = image.astype(np.float64, copy=True)
        if tie_strategy == 'noise' or hybrid_extra_step:
            im_sort += np.random.uniform(-noise_level, noise_level, size=image.shape)
            OA = [1.0] * n_channels
    elif tie_strategy == "none":
        im_sort, OA = image, [1.0] * n_channels

    # --- Histogram mapping stage ---
    im_out = apply_histogram_mapping(im_sort, target_hist, binary_mask)
    im_out[binary_mask==0] = image[binary_mask==0]
    return im_out, OA


def apply_histogram_mapping(
        image: np.ndarray,
        target_hist: np.ndarray,
        binary_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Map pixel ranks to discrete intensity levels to match a target histogram.

    Args:
        image (np.ndarray): Input image with unique-valued pixels (after ordering).
        target_hist (np.ndarray): Target histogram of shape (n_bins, C).
        binary_mask (Optional[np.ndarray]): Optional boolean mask.

    Returns:
        np.ndarray: Histogram-specified image.
    """
    im = im3D(image)
    mask = im3D(binary_mask) if binary_mask is not None else np.ones_like(im, dtype=bool)

    n_bins, C = target_hist.shape
    H, W, C = im.shape
    out = im.copy()

    for c in range(C):
        vals = im[:, :, c][mask[:, :, c]].ravel()
        n_pix = vals.size
        if n_pix == 0:
            continue

        order = np.argsort(vals, kind="mergesort")
        th_sum = target_hist[:, c].sum()
        desired = target_hist[:, c] / th_sum * n_pix

        base = np.floor(desired).astype(np.int64)
        rem = n_pix - base.sum()

        if rem != 0:
            frac = desired - base
            add_idx = np.argsort(-frac)[:abs(rem)]
            base[add_idx] += np.sign(rem)

        levels = np.repeat(np.arange(n_bins), base)
        assign = np.empty_like(vals)
        assign[order] = levels
        out[..., c][mask[..., c]] = assign

    return out


def floyd_steinberg_dithering(image: np.ndarray, depth: int = 256, legacy_mode: bool = False) -> np.ndarray:
    """
    Implements the dithering algorithm presented in :
        R.W. Floyd, L. Steinberg, An adaptive algorithm for spatial grey scale.
        Proceedings of the Society of Information Display 17, 75Ð77 (1976).

    Args:
        image (np.ndarray): An image of floats ranging from 0 to 1.
        depth (optional) : The number of gray shades. (Default = 256)
        legacy_mode (bool) : If True, uses Matlab's rounding algorithm.

    Returns:
        processed_image (np.ndarray): image matrix containing integer values [1, depth], indicating which luminance value should be used for every pixel.
            Output uses the smallest integer dtype that fits all values.
    """
    if not isinstance(image, np.ndarray) or np.issubdtype(image.dtype, np.integer):
        raise TypeError('image should be a np.ndarray of floats ranging from 0 to 1')
    if not isinstance(depth, int):
        raise TypeError('depth should be an integer')

    tim = image * (depth - 1.0)
    for xx in np.arange(1,image.shape[1]-1,1):
        for yy in np.arange(1,image.shape[0]-1,1):  # exchange with the following
            old_pixel = tim[yy,xx]
            new_pixel = MatlabOperators.round(tim[yy,xx]) if legacy_mode else np.round(tim[yy,xx])
            quant_error = old_pixel - new_pixel
            tim[yy,xx+1] = tim[yy,xx+1] + 7/16 * quant_error
            tim[yy+1,xx-1] = tim[yy+1,xx-1] + 3/16 * quant_error
            tim[yy+1,xx] = tim[yy+1,xx] + 5/16 * quant_error
            tim[yy+1,xx+1] = tim[yy+1,xx+1] + 1/16 * quant_error

    c_tim = np.clip(tim, 0, depth-1)

    r_tim = MatlabOperators.round(tim) if legacy_mode else np.round(tim)
    rc_tim = np.rint(np.clip(r_tim, 0, depth-1))
    uint_image = rc_tim.astype(np.min_scalar_type(int(rc_tim.max()))).squeeze()
    return uint_image


def soft_clip(arr: np.ndarray,
              min_value: float = 0.0,
              max_value: float = 1.0,
              max_percent: float = 0.05,
              tol: float = 1e-4,
              verbose: bool = True) -> np.ndarray:
    """
    Softly clip an array to [min_value, max_value] while ensuring that
    the proportion of clipped values does not exceed `max_percent`.

    If naive clipping would clip more than `max_percent` of values,
    the function rescales the array to reduce clipping until the
    clipped proportion is approximately `max_percent`.

    Args:
        arr (np.ndarray): Input array.
        min_value (float): Minimum allowed value after clipping.
        max_value (float): Maximum allowed value after clipping.
        max_percent (float): Maximum allowed proportion (0–1) of clipped values.
        tol (float): Optimization stops early if the clipped proportion is within `tol` (default 1e-4) of target
        verbose (bool): If True, print diagnostic information during processing.

    Returns:
        np.ndarray: Clipped (and possibly rescaled) array.
    """

    def _zero_clip_mean_preserving(arr, a, b):
        x_min = np.min(arr)
        x_max = np.max(arr)
        m = np.mean(arr, dtype=np.float64)
        # If mean is out of bounds, cannot avoid clipping without shifting
        if not (a <= m <= b):
            return None  # signal fallback to affine
        s_up = np.inf if x_max == m else (b - m) / (x_max - m)
        s_down = np.inf if x_min == m else (m - a) / (m - x_min)
        s_star = min(1.0, s_up, s_down)
        return m + s_star * (arr - m)

    if max_percent == 0:
        print(f'{Bcolors.WARNING}[soft_clip] Special case: 0% clipping allowed.{Bcolors.ENDC}')
        return _zero_clip_mean_preserving(arr, min_value, max_value)

    max_iter = 50

    # Flatten for clipping proportion calculation
    flat = arr.ravel()
    total = flat.size

    # --- Step 1: Naive clipping check ---
    below = np.sum(flat < min_value)
    above = np.sum(flat > max_value)
    clipped_fraction = (below + above) / total

    if verbose:
        print(f"[soft_clip] Naive clipping would affect "
              f"{Bcolors.OKBLUE}{clipped_fraction*100:.3f}%{Bcolors.ENDC} of values")

    if clipped_fraction <= max_percent:
        return np.clip(arr, min_value, max_value)

    # --- Step 2: Rescale to match target ---
    arr_min, arr_max = np.min(flat), np.max(flat)
    arr_range = arr_max - arr_min
    if arr_range == 0:
        if verbose:
            print("[soft_clip] Constant array detected — nothing to clip.")
        return np.clip(arr, min_value, max_value)

    target = max_percent
    lo, hi = 0.0, 1.0
    mean_val = np.mean(flat)
    scaled = arr.copy()

    for i in range(max_iter):
        mid = (lo + hi) / 2
        scaled_flat = (flat - mean_val) * mid + mean_val

        below = np.sum(scaled_flat < min_value)
        above = np.sum(scaled_flat > max_value)
        frac = (below + above) / total

        if verbose:
            print(f"[soft_clip] Iter {i:02d}: scale={mid:.5f}, clipped="
                  f"{Bcolors.OKBLUE}{frac*100:.3f}%{Bcolors.ENDC}")

        # Early stopping if we're close enough to target
        if abs(frac - target) <= tol:
            scaled = scaled_flat.reshape(arr.shape)
            break

        if frac > target:
            hi = mid  # shrink more
        else:
            lo = mid  # allow more
            scaled = scaled_flat.reshape(arr.shape)

    return np.clip(scaled, min_value, max_value)


def noisy_bit_dithering(image: np.ndarray, depth: int = 256, legacy_mode: bool = False) -> np.ndarray:
    """
    Implements the dithering algorithm presented in :
        Allard, R., Faubert, J. (2008) The noisy-bit method for digital displays:
        converting a 256 luminance resolution into a continuous resolution. Behavior
        Research Method, 40(3), 735-743.

    Args:
        image (np.ndarray): An image of floats ranging from 0 to 1.
        depth (optional) : The number of gray shades. (Default = 256)
        legacy_mode (bool) : If True, uses Matlab's rounding algorithm.

    Returns:
        processed_image (np.ndarray): image matrix containing integer values [1, depth], indicating which luminance value should be used for every pixel.
            Output uses the smallest integer dtype that fits all values.
    E.g.:
        processed_image = noisy_bit_dithering(image, depth = 256)

    This example assumes that all rgb values are linearly related to luminance
    values (e.g. on a Mac, put your LCD monitor gamma parameter to 1 in the Displays
    section of the System Preferences). If this is not the case, use a lookup table
    to transform the tim integer values into rgb values corresponding to evenly
    spaced luminance values.

    Frederic Gosselin, 27/09/2022
    frederic.gosselin@umontreal.ca

    Slight modifications for Matlab compatibility: Nicolas Dupuis-Roy & Mathias Salvas-Hébert, 2025-08-19

    """
    if not isinstance(image, np.ndarray) or np.issubdtype(image.dtype, np.integer):
        raise TypeError('image should be a np.ndarray of floats ranging from 0 to 1')
    if not isinstance(depth, int):
        raise TypeError('depth should be an integer')

    processed_image = image * (depth - 1.0)

    # tim = np.uint8(np.fmax(np.fmin(np.around(tim + np.random.random(np.shape(im)) - 0.5), depth - 1.0), 0.0))
    processed_image = processed_image + np.random.random(np.shape(image)) - 0.5
    c_tim = np.clip(processed_image, 0, depth-1)
    rc_tim = MatlabOperators.round(c_tim) if legacy_mode else np.rint(c_tim)

    uint_image = rc_tim.astype(np.min_scalar_type(int(rc_tim.max()))).squeeze()
    return uint_image


def uint_to_float01(image: np.ndarray, apply_clipping: bool = True) -> np.ndarray:
    """
    Convert an N-bit unsigned integer (uintN) image to a floating-point image with values ranging from 0 to 1.

    A float is assumed to be within the [0, 1] range whereas the uintN within the [0, n_levels] range.

    Args:
        image (np.ndarray): Input image as a NumPy array with floating-point values.
        apply_clipping (bool): If True, clip values outside the range [0, 1].
                               If False, raises an error if values are out of range.

    Returns:
        np.ndarray: The converted image as a NumPy array with dtype float64.

    Raises:
        ValueError: If `apply_clipping` is False and the image contains values outside the range [0, 1].
    """
    if not isinstance(image, np.ndarray) or not np.issubdtype(image.dtype, np.integer):
        raise TypeError('image should be a np.ndarray of integers')

    # Determine the range of the input image
    image_min, image_max = image.min(), image.max()

    # Scale image if range is [0, 1]
    n_levels = np.iinfo(image.dtype).max
    image = image.astype(np.float64) / n_levels

    # Check if values are within [0, 255]
    if not apply_clipping and (image_min < 0 or image_max > 1):
        raise ValueError("Image contains values outside the range [0, 1]. Consider enabling clipping.")

    # Clip values if allowed
    image_clipped = np.clip(image, 0, 1) if apply_clipping else image

    # Convert to uint8
    return image_clipped.astype(np.float64)


def float01_to_uint(image: np.ndarray, apply_clipping: bool = True, apply_rounding: bool = True, bit_size: int = 8, verbose: bool = True) -> np.ndarray:
    """
    Convert a floating-point image to an n-bit unsigned integer (uintN) image.

    A float is assumed to be within the [0, 1] range whereas the uintN within the [0, n_levels] range.

    Args:
        image (np.ndarray): Input image as a NumPy array with floating-point values.
        apply_clipping (bool): If True, clip values outside the range [0, n_levels].
                               If False, raises an error if values are out of range.
        apply_rounding (bool): If True, round values using np.rint
        bit_size (int): Bit size of the unsigned integer.
        verbose (bool): Warn if clipping needed.

    Returns:
        np.ndarray: The converted image as a NumPy array with dtype uintN.

    Raises:
        ValueError: If `apply_clipping` is False and the image contains values outside the range [0, n_levels].
    """

    if not isinstance(image, np.ndarray) or not np.issubdtype(image.dtype, np.floating):
        return image

    # Define a mapping from bit_size to NumPy unsigned integer types
    uint_types = {
        8: np.uint8,
        16: np.uint16,
        32: np.uint32,
        64: np.uint64,
    }

    if bit_size not in uint_types:
        raise ValueError(f"Invalid bit size {bit_size}. Supported values are {list(uint_types.keys())}.")

    # Get the target dtype
    target_dtype = uint_types[bit_size]

    # Clip the input image to fit within the range of the target dtype
    dtype_info = np.iinfo(target_dtype)

    n_levels = dtype_info.max

    # Check if values are within [0, 1]
    mn, mx = image.min(), image.max()
    if not apply_clipping and (mn < 0 or mx > 1):
        raise ValueError(f"Image contains values outside the range [0, {n_levels}]. Consider enabling clipping.")
    if verbose and (mn < 0 or mx > n_levels):
        txt = 'Values will' + (' ' if apply_clipping else ' not ') + 'be clipped.'
        print(f'{Bcolors.WARNING}Out of range values: Actual range [{mn}, {mx}] outside of the admitted range [0, 1]\n{txt}{Bcolors.ENDC}')

    # Clip values if allowed
    image_clipped = np.clip(image, 0, 1) if apply_clipping else image
    image_clipped = image_clipped * n_levels

    # Round values if allowed
    image_rounded = np.rint(image_clipped) if apply_rounding else image_clipped

    # Convert to uintX
    return image_rounded.astype(target_dtype)


def pol2cart(magnitude: np.ndarray, angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert polar coordinates (magnitude, angle) to Cartesian coordinates (x, y).

    Args:
        magnitude (np.ndarray): Distance from the origin (radius).
        angle (np.ndarray): Angle in radians.

    Returns:
        tuple: A tuple (x, y) where:
            - x is the Cartesian x-coordinate.
            - y is the Cartesian y-coordinate.
    """
    x = magnitude * np.cos(angle)  # Compute x-coordinate
    y = magnitude * np.sin(angle)  # Compute y-coordinate
    return x, y


def cart2pol(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (magnitude, angle).

    Args:
        x (np.ndarray): Real part (x-coordinate).
        y (np.ndarray): Imaginary part (y-coordinate).

    Returns:
        tuple: A tuple (magnitude, angle) where:
            - magnitude is the distance from the origin.
            - angle is the angle in radians.
    """
    magnitude = np.sqrt(x**2 + y**2)  # Compute magnitude
    angle = np.arctan2(y, x)          # Compute angle
    return magnitude, angle


def separate(mask: np.ndarray, background: Union[int, float] = 0, background_operator: Literal['!=', '==', '>=', '<=', '>', '<', '!='] = '==', smoothing: bool = False, show_figure: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Function for simple figure-ground segregation.
    Args:
        mask (np.ndarray): Source mask. Could be an image or a bit mask.
        background (Optional[Union[uint8, float64]]); uint8 value of the background ([0,255]) (e.g., 255)
            or float64 value of the background ([0,1]) (e.g., 1); if equals to 300 or not specified, it is the
            value that occurs the most frequently in mask.
        background_operator (Literal['!=', '==', '>=', '<=', '>', '<', '!=']):
            Foreground is pixel values `background_operator` `background`.
            Example: If `background_operator` is '>=' then the foreground = pixels >= background
        smoothing (bool): If true, applies median blur on mask.
        show_figure (bool): If true, shows the foreground and background

    Returns:
        mask_fgr (np.ndarray[bool]): 2D matrix of the same size as the source mask; Foreground is True
            and background is False.
        mask_bgr (np.ndarray[bool]): 2D matrix of the same size as the source mask; Background is True
            and foreground is False.
        background (Optional[np.uint8]): Specifies the value that was used to
            define the background in the original image

    """

    mask = im3D(mask)
    mask = mask.astype(np.float64)/255 if np.max(mask) > 1 else mask
    background = background/255 if 1 < background < 256 else background

    if background == 300:
        # Use np.unique to get unique values and their counts
        unique_values, counts = np.unique(mask.flatten(), return_counts=True)
        background = unique_values[np.argmax(counts)]

    mask_fgr = np.ones(mask.shape)
    mask_bgr = np.zeros(mask.shape)
    if background_operator == '==':
        mask_bgr = mask == background
    elif background_operator == '!=':
        mask_bgr = mask == background
    elif background_operator == '>':
        mask_bgr = mask > background
    elif background_operator == '>=':
        mask_bgr = mask >= background
    elif background_operator == '<':
        mask_bgr = mask < background
    elif background_operator == '<=':
        mask_bgr = mask <= background

    # Apply median filter to smooth the mask
    if mask_bgr.mean() != 1:
        if smoothing:
            mask_bgr = apply_median_blur(np.uint8(mask_bgr))
        mask_fgr = (mask_bgr * -1) + 1
    else:
        raise ValueError('All pixels are masked!')

    if show_figure:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(mask_fgr, cmap='gray')
        plt.title('Foreground Mask')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_bgr, cmap='gray')
        plt.title('Background Mask')
        plt.show()

    return mask_fgr.astype(bool), mask_bgr.astype(bool), background


def image_spectrum(image: np.ndarray, rescale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectrum of an image
    Args:
        image (np.ndarray): An image
        rescale (bool): If true, will rescale each channel to [0, 1] range.

    Returns:
        magnitude, phase

    TODO: Optimization: Check image type and use np.fft.rfft2 for faster computations.
    """
    image = im3D(image)
    if rescale:
        image = rescale_image(image, 0, 1)  # [0, 255] -> [0, 1]

    x_size, y_size, n_channels = image.shape
    phase = np.zeros((x_size, y_size, n_channels))  # Phase FT
    magnitude = np.zeros((x_size, y_size, n_channels))  # Magnitude FT

    image = im3D(image)
    for channel in range(image.shape[-1]):
        fft_image = np.fft.fftshift(np.fft.fft2(image[:, :, channel]))
        magnitude[:, :, channel], phase[:, :, channel] = cart2pol(np.real(fft_image), np.imag(fft_image))
    return magnitude, phase


def gaussian_kernel(
    size: int,
    sigma: Optional[float] = None,
    coverage: Optional[float] = None,
    n_dim: int = 2
) -> np.ndarray:
    """Generate a normalized Gaussian kernel.

    Exactly one of `sigma` or `coverage` must be provided.

    If `coverage` is specified, `sigma` is automatically computed so that
    the specified fraction of the Gaussian's total area is contained
    within the kernel support of given `size`.

    Args:
        size (int): Size of the kernel (must be odd).
        sigma (float, optional): Standard deviation of the Gaussian.
            Mutually exclusive with `coverage`.
        coverage (float, optional): Fraction (0–1) of total Gaussian area
            contained within the kernel (e.g., 0.99 ≈ ±2.575σ).
            Mutually exclusive with `sigma`.
        n_dim (int, optional): Dimensionality of the kernel.
            * 1 -> return 1D Gaussian kernel of shape (size,)
            * 2 -> return 2D Gaussian kernel of shape (size, size)
            Defaults to 2.

    Returns:
        np.ndarray: Normalized Gaussian kernel of shape (size,) or (size, size).

    Raises:
        ValueError: If `size` is not odd, or if both/neither of `sigma` and
            `coverage` are provided, or if inputs are invalid.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if (sigma is None and coverage is None) or (sigma is not None and coverage is not None):
        raise ValueError("Provide exactly one of `sigma` or `coverage`.")
    if n_dim not in (1, 2):
        raise ValueError("n_dim must be 1 or 2.")

    # Determine sigma from coverage
    if coverage is not None:
        if not (0 < coverage < 1):
            raise ValueError("`coverage` must be between 0 and 1.")
        half_width = size // 2
        p = (1 + coverage) / 2.0
        # Gaussian inverse CDF = sqrt(2) * erfinv(2p - 1)
        x = 2 * p - 1
        erfinv_x = _erfinv_approx(x)
        n_sigma = np.sqrt(2) * erfinv_x
        sigma = half_width / n_sigma

    # Coordinate axis centered at 0
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)

    if n_dim == 1:
        g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    # n_dim == 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def _erfinv_approx(x: np.ndarray) -> np.ndarray:
    """Approximate inverse error function (Winitzki’s formula).
    Valid for |x| < 1; ~1e-6 accuracy for typical use."""
    a = 0.147
    ln_term = np.log(1 - x**2)
    first = 2 / (np.pi * a) + ln_term / 2
    second = ln_term / a
    return np.sign(x) * np.sqrt(np.sqrt(first**2 - second) - first)


def center_surround_kernel(
    size: int,
    sigma_center: Optional[float] = None,
    coverage: Optional[float] = None,
    ratio: float = 1.6,
    n_dim: int = 2
) -> np.ndarray:
    """Generate a center–surround (Difference-of-Gaussians) kernel.

    The kernel is built as a narrow (center) Gaussian minus a broader (surround)
    Gaussian, each L1-normalized over the *same* finite support defined by
    `size`. The resulting kernel is DC-balanced (sum≈0), suitable for
    center–surround/edge-like filtering and strict-ordering feature banks.

    Exactly one of `sigma_center` or `coverage` must be provided.

    If `coverage` is provided, `sigma_center` is chosen so that the specified
    fraction of the 1D Gaussian's total area lies within the half-width
    `size//2`. `sigma_surround` is then derived as `ratio * sigma_center`.

    Args:
        size: Odd kernel size; defines spatial support (1D or 2D).
        sigma_center: Standard deviation of the center Gaussian (pixels).
        coverage: Fraction (0–1) of total Gaussian area within support, used to
            *infer* `sigma_center`. (E.g., 0.95 ≈ ±2σ, 0.99 ≈ ±3σ.)
        ratio: Surround-to-center sigma ratio. Common choices:
            - 1.6 → good LoG approximation / Marr–Hildreth/SIFT-style           (default)
            - ~3–6 → retinal ganglion center–surround (HVS-like)
        n_dim: 1 → shape (size,), 2 → shape (size, size).

    Returns:
        DoG kernel (float64) of shape (size,) or (size, size), zero-mean.

    Raises:
        ValueError: On invalid sizes, exclusivity of args, or nonpositive sigmas.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if (sigma_center is None) == (coverage is None):
        raise ValueError("Provide exactly one of `sigma_center` or `coverage`.")
    if n_dim not in (1, 2):
        raise ValueError("`n_dim` must be 1 or 2.")
    if sigma_center is not None and sigma_center <= 0:
        raise ValueError("`sigma_center` must be positive.")
    if ratio <= 1.0:
        raise ValueError("`ratio` must be > 1.0 (surround broader than center).")

    # Infer sigma_center from coverage, matching your gaussian_kernel convention
    if coverage is not None:
        if not (0 < coverage < 1):
            raise ValueError("`coverage` must be in (0, 1).")
        half_width = size // 2
        p = (1.0 + coverage) / 2.0
        n_sigma = np.sqrt(2.0) * _erfinv_approx(2 * p - 1)
        sigma_center = half_width / n_sigma

    sigma_surround = ratio * float(sigma_center)

    # Coordinate grid
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)

    if n_dim == 1:
        g_c = np.exp(-(ax**2) / (2 * sigma_center**2))
        g_s = np.exp(-(ax**2) / (2 * sigma_surround**2))
        g_c /= g_c.sum()
        g_s /= g_s.sum()
        dog = g_c - g_s
        dog -= dog.mean()  # enforce zero DC over finite support
        return dog

    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    g_c = np.exp(-r2 / (2 * sigma_center**2))
    g_s = np.exp(-r2 / (2 * sigma_surround**2))
    g_c /= g_c.sum()
    g_s /= g_s.sum()
    dog = g_c - g_s
    dog -= dog.mean()
    return dog


def laplacian_kernel(
    size: int,
    sigma: Optional[float] = None,
    coverage: Optional[float] = None,
    n_dim: int = 2
) -> np.ndarray:
    """Generate a Laplacian-of-Gaussian (LoG) kernel over finite support.

    Exactly one of `sigma` or `coverage` must be provided.

    If `coverage` is given, `sigma` is chosen so the specified fraction of the
    1D Gaussian area lies within half-width `size//2` (matching the convention
    used in `gaussian_kernel`).

    Args:
        size: Odd kernel size; defines spatial support (1D or 2D).
        sigma: Standard deviation of Gaussian envelope (pixels).
        coverage: Fraction (0–1) of total Gaussian area within support.
        n_dim: 1 → shape (size,), 2 → shape (size, size).

    Returns:
        Zero-mean LoG kernel (float64) of shape (size,) or (size, size).

    Raises:
        ValueError: On invalid sizes, exclusivity of args, or nonpositive sigma.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if (sigma is None) == (coverage is None):
        raise ValueError("Provide exactly one of `sigma` or `coverage`.")
    if n_dim not in (1, 2):
        raise ValueError("`n_dim` must be 1 or 2.")

    if coverage is not None:
        if not (0 < coverage < 1):
            raise ValueError("`coverage` must be in (0, 1).")
        half_width = size // 2
        p = (1.0 + coverage) / 2.0
        n_sigma = np.sqrt(2.0) * _erfinv_approx(2 * p - 1)
        sigma = half_width / n_sigma

    if sigma <= 0:
        raise ValueError("`sigma` must be positive.")

    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)

    if n_dim == 1:
        x2 = ax**2
        g = np.exp(-x2 / (2 * sigma**2))
        log1d = (x2 - sigma**2) / (sigma**4) * g
        log1d -= log1d.mean()
        return log1d

    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    g = np.exp(-r2 / (2 * sigma**2))
    log2d = (r2 - 2 * sigma**2) / (sigma**4) * g
    log2d -= log2d.mean()
    return log2d


def tie_breaking_noise_level(image: np.ndarray, min_gap: Optional[float] = None, gap_frac_cap: float = 1e-3) -> float:
    """Return a numerically effective yet rank-safe noise amplitude for any dtype.

    The returned noise is large enough to survive rounding of the input dtype
    (unit-in-the-last-place (ULP)-aware for floats, quantization-aware for uint-x) but small enough not
    to reorder distinct values. Used to ensure strict ranking in exact
    histogram specification when pixel ties are present.

    Args:
        image (np.ndarray): Input image. Supported dtypes: uint8, uint16,
            float16, float32, float64.
        min_gap (float, optional): Smallest nonzero intensity increment
            in the data (e.g., from np.diff(np.unique(image))). If provided,
            the returned noise will be capped to a small fraction of it.
        gap_frac_cap (float): Fraction of `min_gap` allowed for noise (default 1e-3).

    Returns:
        float: Recommended amplitude of uniform tie-breaking noise (symmetric ±).
    """
    x = np.asarray(image)

    # --- Determine dtype category ---
    if np.issubdtype(x.dtype, np.floating):
        finfo = np.finfo(x.dtype)
        eps = finfo.eps
        max_abs = float(np.max(np.abs(x)))
        rng = float(np.max(x) - np.min(x))

        # dtype-dependent safety parameters
        if x.dtype == np.float64:
            ulp_factor = 10.0
            rel_range = 1e-9
        elif x.dtype == np.float32:
            ulp_factor = 10.0
            rel_range = 1e-6
        elif x.dtype == np.float16:
            ulp_factor = 5.0
            rel_range = 1e-3
        else:
            # fallback for exotic float types (e.g. bfloat16)
            ulp_factor = 10.0
            rel_range = 1e-6

        # (1) ULP-based floor: ensures the noise survives rounding
        ulp_based = ulp_factor * eps * max_abs
        # (2) Range-based cap: ensures noise stays negligible
        range_based = rel_range * max(rng, 1.0)
        noise = max(ulp_based, range_based)

    elif np.issubdtype(x.dtype, np.integer):
        # Handle integer types (e.g. uint8, uint16)
        info = np.iinfo(x.dtype)
        # one quantization step is 1 intensity unit
        quant_step = 1.0
        # add noise smaller than 1 LSB to avoid rank distortion
        noise = 1e-3 * quant_step  # e.g. 0.001 for 8-bit, 0.001 for 16-bit
    else:
        raise TypeError(f"Unsupported image dtype: {x.dtype}")

    # --- Optionally cap by the observed smallest gap ---
    if min_gap is not None and np.isfinite(min_gap) and min_gap > 0.0:
        noise = min(noise, gap_frac_cap * min_gap)

    return noise


def print_log(logs: List[str], log_path: Union[Path, str], log_name: Optional[str] = None) -> None:
    """
    Takes a list of log messages and writes them to a file located
    at the specified log directory (`log_path`). Optionally, a static filename can
    be provided (`log_name`). If no static name is supplied, the function generates
    a filename containing the current date and time, ensuring logs are uniquely
    stored based on their creation time.

    Args:
        logs (List[str]): A list of log messages to write to the file.
        log_path (Path): The directory where the log file will be saved.
        log_name (Optional[str]): The optional static name for the log file. If not
            specified, a timestamped filename will be generated.

    Returns:
        None
    """
    if not isinstance(logs, Iterable):
        raise TypeError('logs must be a list of string.')

    if not isinstance(log_path, (Path, str)):
        raise TypeError('log_path must be a Path object or a string')
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileExistsError(f'{log_path} does not exists.')

    # Generate a filename with the full date and time
    if log_name is None:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = f"log_{current_datetime}.txt"
    else:
        if 'txt' not in log_name:
            raise ValueError(f'log_name ({log_name}) must be a text file with a .txt extension.')

    filename = Path(log_path) / log_name

    # Write each log to a new line in the file
    with open(filename, 'w') as file:
        for log in logs:
            file.write(strip_ansi(log) + '\n')


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def colorize(text: str, color: str) -> str:
    """Return text wrapped in ANSI color codes."""
    return f"{color}{text}{Bcolors.ENDC}"


def console_log(msg: str, indent_level: int = 0, color: Optional[str] = None, verbose: bool = True, strip: bool = True) -> str:
    """
    Logs a message to the console with optional indentation, color, and verbose control.

    Args:
        msg (str): The message string to be logged.
        indent_level (int): The level of indentation represented as the number of tab characters.
            Defaults to 0.
        color (Optional[str]): The color code applied to the message text.
            Defaults to None, indicating no color formatting.
        verbose (bool): A flag to determine whether to print the message to the console.
            If False, the message is only processed and not output. Defaults to False.
        strip (bool): String ainsi characters

    Returns:
        str: The formatted message as a string with any ANSI color codes stripped.
    """

    def _set_indent_and_color(text, lev: int, col: Optional[str] = None):
        indent_str = '\t' * lev
        if col is not None:
            return "\n".join(f'{indent_str}{col}{line}{Bcolors.ENDC}' for line in text.splitlines())
        else:
            return "\n".join(f'{indent_str}{line}' for line in text.splitlines())

    def _coat_check(
            text: str,
            transform: Callable[[str], str]) -> str:
        prefix_parts: list[str] = []
        remaining = text
        ESC_UP_ONE = "\x1b[F"

        while remaining:
            if remaining.startswith(ESC_UP_ONE):
                prefix_parts.append(ESC_UP_ONE)
                remaining = remaining[len(ESC_UP_ONE):]
            elif remaining.startswith("\n"):
                prefix_parts.append("\n")
                remaining = remaining[1:]
            else:
                break

        # --- 2) Strip & record trailing newlines ---
        trailing_count = 0
        # Work on a separate variable so we don't lose remaining entirely
        tmp = remaining
        while tmp.endswith("\n"):
            trailing_count += 1
            tmp = tmp[:-1]

        core = tmp
        suffix_newlines = "\n" * trailing_count

        # --- 3) Transform the core text ---
        transformed_core = transform(core)

        # --- 4) Reassemble ---
        prefix = "".join(prefix_parts)
        return prefix + transformed_core + suffix_newlines

    # Log message
    msg = _coat_check(msg, lambda core: _set_indent_and_color(core, indent_level, color))
    # msg = _set_indent_and_color(msg, indent_level, color)
    if verbose:
        print(msg)
    if strip:
        msg = strip_ansi(msg)
    return msg


def show_processing_overview(processor: ImageProcessor, img_idx: int = 0, show_figure: bool = True, show_initial_target: bool = False) -> plt.Figure:
    """Display before/after images and diagnostics for all processing steps in one figure.

    The figure layout adapts to the active SHINIER mode:
        • Row 1: before/after images.
        • Subsequent rows: one row per processing step (e.g., luminance, histogram, spectrum).
          Each diagnostic row shows "before" (left) and "after" (right) panels side by side.

    Args:
        processor (ImageProcessor): The SHINIER ImageProcessor instance.
        img_idx (int, optional): Index of the image to visualize. Defaults to 0.
        show_figure (bool): If False, return the fig object without showing it (i.e. plt.show())
        show_initial_target (bool): If True, plots the initial target in composite modes.

    Returns:
        matplotlib.figure.Figure: Composite figure summarizing the image transformations.
    """

    import os, matplotlib
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("TkAgg")

    fontname = 'Arial'

    # --- Retrieve relevant info ---
    steps = getattr(processor, "_processing_steps", [])
    n_steps = len(steps)
    name_map = getattr(processor, "_fct_name2process_name", {})

    mode = getattr(processor.options, "mode", None)
    with Image.open(processor.dataset.images.src_paths[img_idx]) as im:
        img_before = np.asarray(im)
    is_rgb_before = img_before.ndim == 3 and img_before.shape[2] == 3
    img_after = processor.dataset.images[img_idx]
    is_rgb_after = img_after.ndim == 3 and img_after.shape[2] == 3
    masks = getattr(processor, "bool_masks", [None])[img_idx]

    # --- Figure layout ---
    diag_steps = [s for s in steps if s != "dithering"]
    n_rows = 1 + len(diag_steps) if len(diag_steps) > 0 else 1
    fig = plt.figure(figsize=(11.5, 3.6 * n_rows))
    gs = GridSpec(
        n_rows, 2, figure=fig,
        height_ratios=[2.0] + [1.5] * (n_rows - 1),
        hspace=0.38, wspace=0.18
    )
    # --- Row 1: Before/After images ---
    ax_before = fig.add_subplot(gs[0, 0])
    ax_after = fig.add_subplot(gs[0, 1])
    ax_before.imshow(img_before, cmap="gray" if not is_rgb_before else None)
    ax_after.imshow(img_after, cmap="gray" if not is_rgb_after else None)
    for ax, title in zip([ax_before, ax_after], ["Before", "After"]):
        ax.set_title(title, fontsize=13, fontname=fontname)
        ax.axis("off")

    # --- Per-step diagnostics ---
    for i, step in enumerate(steps, start=1):
        readable = name_map.get(step, step)
        axL = fig.add_subplot(gs[i, 0])
        axR = fig.add_subplot(gs[i, 1])

        # ---- Luminance matching ----
        if step == "lum_match":
            _ = imhist_plot(
                img=processor._initial_buffer[img_idx],
                binary_mask=masks if masks is not None else None,
                descriptives=True,
                title=f"Before – {readable}",
                ax=axL
            )
            _ = imhist_plot(
                img=processor._final_buffer[img_idx],
                binary_mask=masks if masks is not None else None,
                descriptives=True,
                title=f"After – {readable}",
                ax=axR
            )

            # Overlay mean/std text box for target comparison
            t_mu, t_sd = getattr(processor, "_target_lum", (None, None))
            if t_mu is not None and t_sd is not None:
                text = f"Target μ={t_mu:.1f}, σ={t_sd:.1f}"
                fig.text(0.72, 0.26 - 0.18 * (i - 1), text, fontsize=9, va="top")

        # ---- Histogram matching ----
        elif step == "hist_match":
            final_target_hist = processor._target_hist
            if 'hist' in processor._initial_targets.keys() and show_initial_target:
                initial_target_hist = processor._initial_targets['hist']
            else:
                initial_target_hist = final_target_hist

            _ = imhist_plot(
                img=processor._initial_buffer[img_idx],
                target_hist=initial_target_hist,
                binary_mask=masks if masks is not None else None,
                descriptives=False,
                title=f"Before – {readable}",
                ax=axL,
                show_normalized_rmse=True
            )
            _ = imhist_plot(
                img=processor._final_buffer[img_idx],
                target_hist=initial_target_hist,
                binary_mask=masks if masks is not None else None,
                descriptives=False,
                title=f"After – {readable}",
                ax=axR,
                show_normalized_rmse=True
            )
            if n_steps > 1 and show_initial_target:
                axR.plot(final_target_hist[:, 0], ls='--', lw=1, color='gray', label=f'Moving target')
                axR.legend(frameon=False, fontsize=9, loc='upper right')

        # ---- Spatial frequency matching ----
        elif step == "sf_match":
            final_target_sf = processor._target_sf ** 2
            if 'sf' in processor._initial_targets.keys() and show_initial_target:
                initial_target_sf = processor._initial_targets['sf'] ** 2
            else:
                initial_target_sf = final_target_sf

            avg_before, radii = sf_profile(
                processor._initial_buffer[img_idx],
                legacy_mode=processor.options.legacy_mode,
            )
            avg_after, radii = sf_profile(
                processor._final_buffer[img_idx],
                legacy_mode=processor.options.legacy_mode
            )
            _ = sf_plot(processor._initial_buffer[img_idx], sf_p=avg_before, target_sf=initial_target_sf, ax=axL, show_normalized_rmse=True)
            _ = sf_plot(processor._final_buffer[img_idx], sf_p=avg_after, target_sf=initial_target_sf, ax=axR, show_normalized_rmse=True)

            if n_steps > 1 and show_initial_target:
                image = im3D(processor._initial_buffer[img_idx])
                xs, ys, channels = image.shape
                R = int(np.floor(min(xs, ys) / 2.0))
                if final_target_sf is not None and final_target_sf.shape[0] > xs / 2:
                    final_target_sf = final_target_sf[1: R + 1]
                axR.loglog(radii[:, 0], final_target_sf[:, 0], ls='--', lw=1, color='gray', label=f'Moving target')
                axR.legend(frameon=False, fontsize=9, loc='upper right')

        # ---- Fourier spectrum matching ----
        elif step == "spec_match":
            target_spectrum = processor._target_spectrum
            mag_before, _ = image_spectrum(processor._initial_buffer[img_idx])
            mag_after, _ = image_spectrum(processor._final_buffer[img_idx])
            _ = spectrum_plot(mag_before, ax=axL, target_spectrum=target_spectrum, show_normalized_rmse=True)
            _ = spectrum_plot(mag_after, ax=axR, target_spectrum=target_spectrum, show_normalized_rmse=True)

        # ---- Dithering ----
        elif step == "dithering":
            # Dithering only affects appearance (already visible in row 1)
            axL.text(0.5, 0.5, "Dithering only", ha="center", va="center")
            axR.axis("off")

    (y0L, y1L) = axL.get_ylim()
    (y0R, y1R) = axR.get_ylim()
    ymin, ymax = min(y0L, y0R), max(y1L, y1R)

    axL.set_ylim(ymin, ymax)
    axR.set_ylim(ymin, ymax)

    fig.suptitle(
        f"SHINIER — Mode {mode}: {', '.join(name_map.get(s, s) for s in steps)}",
        fontsize=15,
        weight="bold",
        fontname="Times New Roman",
        y=0.99,
    )
    if show_figure:
        fig.show()

    return fig


def beta_bounds_from_ssim(gradients: np.ndarray, ssim: List[float], binary_mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute Avanaki's step-size bounds for SSIM gradient ascent.

    This computes lower/upper bounds for the scalar step size β in an update
    of the form: ``Y <- Y + β * (M ⊙ ∇_Y SSIM(I, Y))``, where ``M`` is an
    optional binary mask.

    Bounds:
      - β_max = (1 - SSIM(I, Y)) / ||G||_2^2
      - β_min = 1 / (2 * ||G||_∞)
    with G the (optionally masked) gradient map.

    Args:
      gradients: Gradient map ∇_Y SSIM(I, Y); shape (H, W) or (H, W, C).
      ssim: List of SSIM(I, Y), one per channel.
      binary_mask: Optional boolean mask of shape (H, W) or broadcastable to (H, W, n_channels).
                   Pixels with False are frozen (no update).

    Returns:
      A list (len = n_channels) of tuples (beta_min, beta_max) giving the lower and upper bounds. Returns zeros
      if the gradient is zero everywhere under the mask or SSIM≈1.

    Notes:
      - See "Iterative exact global histogram specification and SSIM gradient ascent: a proof of convergence, step size and parameter selection"
      - Norms are computed over all pixels and channels after masking.
      - Gradients should already include the same scaling used in your SSIM implementation.
    """
    bm = im3D(binary_mask)
    G = im3D(gradients)
    n_channels = G.shape[2]
    if bm.shape != G.shape:
        raise ValueError('`gradients` and `binary_mask` should be of equal shape.')
    if len(ssim) != n_channels:
        raise ValueError('Length of `ssim` should be equal to number of channel(s) in `gradients` and `binary_mask`')

    out = []
    for ch in range(n_channels):
        g = G[..., ch][bm[..., ch]]
        N = bm[..., ch].sum()

        # Clean numerics
        if not np.all(np.isfinite(g)):
            g = np.nan_to_num(g, copy=False)

        # Norms: sum of squares for L2^2; max abs for L_inf
        g2 = float(np.dot(g, g))  # Same as np.sum(g**2)
        g_inf = float(np.max(np.abs(g))) if g.size else 0.0
        s = float(ssim[ch])

        if g2 <= 0.0 or g_inf <= 0.0 or s >= 1.0:
            # Stationary/fully masked or already at SSIM=1: no meaningful step.
            beta_max, beta_min = 0, 0
        else:
            beta_max = (1.0 - s) / g2 / N
            beta_min = 1.0 / (2.0 * g_inf) / N
        out.append((beta_min, beta_max))
    return out


def ssim_sens(image1: np.ndarray, image2: np.ndarray, data_range: Optional[float] = None, use_sample_covariance: bool = False, binary_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Structural Similarity Index (SSIM) and its gradient.

    Args:
        image1 (np.ndarray): First image as a 3D array.
        image2 (np.ndarray): Second image as a 3D array.
        data_range (int, optional): Dynamic range of pixel values.
        use_sample_covariance (bool): If True, use sample covariance when computing SSIM.
            - Note that Avanaki (2009) and Wang et al. (2004) used population covariance.
        binary_mask (np.ndarray): Binary mask used to zero-out all masked regions and normalize accordingly.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Gradient of SSIM (sensitivity) as a 2D array.
            - Mean SSIM value as a float.

    References :
        1. Avanaki, A.N. Exact global histogram specification optimized for structural similarity.
        OPT REV 16, 613–621 (2009). https://doi.org/10.1007/s10043-009-0119-z

        2. Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment:
        from error visibility to structural similarity," in IEEE Transactions on Image Processing,
        vol. 13, no. 4, pp. 600-612, April 2004, doi: 10.1109/TIP.2003.819861.

    Notes:
        - Should match scikit-image ssim computations using data_range=255, channel_axis=-1, win_size=11, gaussian_weights=True
    """
    # Keep original dtype for data_range inference
    orig_dtype = image1.dtype

    # Coerce to float64 for stable numerics and force channels-last with your helper
    # Assumes `im3D` is available in your codebase and returns HxWxC (C>=1).
    img1_3D = im3D(image1)
    img2_3D = im3D(image2)
    binary_mask = np.ones(img1_3D.shape, dtype=bool) if binary_mask is None else im3D(binary_mask)

    # ---- Basic checks ----
    if img1_3D.shape != img2_3D.shape:
        raise ValueError("image1 and image2 must have the same shape")
    if binary_mask.shape != img2_3D.shape:
        raise ValueError("binary_mask and image1 and image2 must have the same shape")

    H, W, C = img1_3D.shape

    # ---- data_range handling (REQUIRED for float images; inferred for integer) ----
    if data_range is None:
        if np.issubdtype(orig_dtype, np.floating):
            # For typical float images in [0,1], pass data_range=1.0 explicitly.
            raise ValueError(
                "For float images, please specify data_range (e.g., 1.0 for [0,1], 255 for uint8-equivalent)."
            )
        # If integer, infer from dtype like skimage
        info = np.iinfo(orig_dtype)
        data_range = float(info.max - info.min)
    R = float(data_range)

    # SSIM defaults parameters
    sigma = 1.5
    truncate = 3.5
    r = int(truncate * sigma + 0.5)  # radius (e.g., 5 for sigma=1.5, truncate=3.5)
    win_size = 2 * r + 1  # 11-tap
    pad = r
    NP = win_size * win_size
    cov_norm = (NP / (NP - 1.0)) if use_sample_covariance else 1.0

    # Build normalized 1D Gaussian kernel (to mimic ndimage.gaussian, mode='reflect')
    x = np.arange(-r, r + 1, dtype=np.float64)
    g1d = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g1d /= g1d.sum()

    # ---- Constants (Wang et al.) ----
    K1, K2 = 0.01, 0.03  # Original proposed values
    # K1, K2 = 0.005, 0.003  # See Avanaki, 2010 in the context of exact histogram spec with SSIM optimization
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # C3 = C2 / 2

    all_sens = []
    all_mssim = []
    for ch in range(C):
        X = img1_3D[:, :, ch]
        Y = img2_3D[:, :, ch]

        # Local means (Gaussian)
        ux = convolve_2d(X, g1d)
        uy = convolve_2d(Y, g1d)

        # Second moments
        uxx = convolve_2d(X * X, g1d)
        uyy = convolve_2d(Y * Y, g1d)
        uxy = convolve_2d(X * Y, g1d)

        # Variances / covariance with sample (Bessel) correction to match skimage default
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        # SSIM components (Wang 2004, Eq. 6)
        # - Luminance factor (L)
        A1 = 2.0 * ux * uy + C1
        B1 = ux**2 + uy**2 + C1
        # L = A1/B1

        # - Contrast factor (C)
        A2 = 2.0 * vxy + C2
        B2 = vx + vy + C2
        # C = A2/B2

        # # - Structural factor (S): No need to be specified if factor L, C and S have the same weight and C3 == C2/2
        # A3 = vxy + C3
        # B3 = vx * vy + C3
        # S = A3/B3
        #
        # # Combine the 3 factors -> SSIM
        # SSIM2 = L * C * S
        D = (B1 * B2)
        SSIM = (A1 * A2) /D

        # Crop a border of width `pad` before averaging (skimage behavior)
        # if pad > 0 and min(*SSIM.shape) > 2 * pad:
        #     SSIM_valid = SSIM[pad:-pad, pad:-pad][binary_mask[...,ch]]
        # else:
        SSIM_valid = SSIM[binary_mask[...,ch]]
        mssim = SSIM_valid.mean(dtype=np.float64)
        all_mssim.append(mssim)

        # Gradient (Avanaki 2009, Eqs. 7–8), filtered with the same Gaussian
        # (1) Luminance–contrast coupling term
        term1 = 2.0 * (A1 / D)  # factor of 2 comes from ∂(2*m_x*m_y)/∂Y
        sens = convolve_2d(term1, g1d) * X

        # (2) Contrast–structure term (negative feedback through Y)
        term2 = -2.0 * (SSIM / B2)  # factor of 2 from ∂(2*σ_xy)/∂Y
        sens += convolve_2d(term2, g1d) * Y

        # (3) Cross interaction term between mean and variance components
        term3 = 2.0 * (ux * (A2 - A1) - uy * (B2 - B1) * SSIM) / D
        sens += convolve_2d(term3, g1d)

        # Normalize to obtain gradient of *mean* SSIM (as in Avanaki’s Eq. 8)
        sens /= np.sum(binary_mask[...,ch])
        sens[binary_mask[...,ch] == False] = 0.0
        all_sens.append(sens)

    sens_out = np.stack(all_sens, axis=-1)
    sens_out = im3D(sens_out)
    ssim_vals = np.asarray(all_mssim, dtype=np.float64)  # per-channel SSIMs

    return sens_out, ssim_vals


class StepSizeController:
    """Three-regime adaptive step-size controller for SSIM optimization.

    Behavior:
      A) If SSIM increases noticeably → accept, keep weight.
      B) If SSIM increases only slightly (stall) → restart with larger weight.
      C) If SSIM decreases → restart with smaller weight.

    Attributes:
        gain_up (float): Multiplier when escaping a stall (default=1.3).
        gain_down (float): Multiplier when correcting overshoot (default=0.6).
        stall_thresh (float): ΔSSIM below which we consider a stall (default=1e-5).
        alpha_min (float): Minimum allowed step size.
        alpha_max (float): Maximum allowed step size.
        max_stall_iter (int): Maximum allowed number of stalled iterations.
    """

    def __init__(
        self,
        gain_up: float = 1.3,
        gain_down: float = 0.6,
        stall_thresh: float = 1e-5,
        alpha_min: float = 1e-4,
        alpha_max: float = 10.0,
        max_stall_iter: int = 5
    ):
        self.gain_up = gain_up
        self.gain_down = gain_down
        self.stall_thresh = stall_thresh
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.max_stall_iter = max_stall_iter

        # Internal state
        self.stall_iter: int = 0
        self.best_ssim: float = -np.inf
        self.best_image = None
        self.best_gradient = None
        self.last_ssim = None
        self.all_ssim = []
        self.restart_reason = ' '

    def update(self, alpha: float, ssim_new: float, Y_new: np.ndarray, gradient_new: np.ndarray) -> Tuple[float, np.ndarray, bool]:
        """Update step size and decide whether to restart or accept.

        Args:
            alpha (float): Current step-size weight.
            ssim_new (float): Current mean SSIM value.
            Y_new (np.ndarray): Current image.
            gradient_new (np.ndarray): Current gradient from ssim_sens()

        Returns:
            (alpha_new, Y_next, restart)
                alpha_new       : Updated step-size weight.
                Y_next          : Either new image or reverted one.
                gradient_next   : Either new gradient or reverted one.
                restart         : Whether a restart is needed.
        """
        restart = False
        self.restart_reason = " "
        done = False
        # rounded_sim = np.round(ssim_new * 1/self.stall_thresh) * self.stall_thresh
        self.all_ssim.append(ssim_new)
        all_ssim = np.asarray(self.all_ssim)
        max_ssim = np.max(all_ssim)
        self.stall_iter = np.count_nonzero(all_ssim == max_ssim)

        # First iteration initialization
        if self.last_ssim is None:
            self.last_ssim = ssim_new
            self.best_ssim = ssim_new
            self.best_image = Y_new.copy()
            self.best_gradient = gradient_new.copy()
            return alpha, ssim_new, Y_new, gradient_new, False, done

        delta_ssim = ssim_new - self.last_ssim

        # Case A: Clear improvement
        if delta_ssim > self.stall_thresh:
            # Significant increase → accept progress
            self.last_ssim = ssim_new
            self.best_ssim = ssim_new
            np.copyto(self.best_image, Y_new)
            np.copyto(self.best_gradient, gradient_new)
            self.restart_reason = "SSIM increased!"

        # Case B: Stalling or negligible improvement
        elif delta_ssim == 0:
        # elif 0 <= delta_ssim <= self.stall_thresh:
            # Restart from best with larger alpha
            alpha = np.clip(alpha * self.gain_up, self.alpha_min, self.alpha_max)
            self.last_ssim = self.best_ssim
            Y_new = self.best_image
            gradient_new = self.best_gradient
            restart = True
            self.restart_reason = "SSIM stalled: rollback"

        # Case C: Regression (SSIM decreased)
        elif delta_ssim < 0:
            # Rollback and reduce alpha
            alpha = np.clip(alpha * self.gain_down, self.alpha_min, self.alpha_max)
            self.last_ssim = self.best_ssim
            Y_new = self.best_image
            gradient_new = self.best_gradient
            restart = True
            self.restart_reason = "SSIM decreased: rollback"

        ssim_new = self.last_ssim
        if self.stall_iter > self.max_stall_iter:
            done = True
        return alpha, ssim_new, Y_new, gradient_new, restart, done


def hist_match_validation(images: ImageListIO, binary_masks: List[np.ndarray], target_hist: Optional[np.ndarray] = None, normalize_rmse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates the histogram matching process by comparing initial histograms of images
    to a target histogram. Uses correlation coefficients and root mean square error
    (RMSE) as metrics for validation.

    Args:
        images (ImageListIO): A list-like object containing images. Each image should
            represent its histogram and dimensional attributes appropriately.
        binary_masks ([np.ndarray]). A list of binary masks with same size as image.
        target_hist (Optional[np.ndarray]). A target histogram to compare against.
        normalize_rmse (bool): Return normalized RMSE using its actual range.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple where the first element is an array
            of correlation coefficients, and the second element is an array of RMS
            values for all images compared against the target histogram.
    """

    target_hist, initial_hist = avg_hist(images=images, binary_masks=binary_masks, normalized=True, n_bins=256, output_list_hist=True)

    # Compute metric
    N = len(initial_hist)
    corr, rms = np.zeros((N,)), np.zeros((N,))
    for idx, a_hist in enumerate(initial_hist):
        corr[idx] = np.corrcoef(a_hist.ravel(), target_hist.ravel())[0, 1]
        if normalize_rmse:
            rms[idx] = normalized_rmse(a_hist.ravel(), target_hist.ravel(), mode='actual range')
        else:
            rms[idx] = compute_rmse(a_hist.ravel(), target_hist.ravel())
    return corr, rms


def sf_match_validation(images: ImageListIO, target_spectrum: Optional[np.ndarray] = None, normalize_rmse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates spectral match between a set of input images by comparing their
    rotational averages of magnitude spectra against a computed target spectrum.
    The function calculates metrics such as correlation coefficients and root
    mean square error (RMSE) to evaluate the quality of the spectral match.

    Args:
        images (ImageListIO): Array of images for which the spectral validation
            is performed. Each image is assumed to have three channels (e.g., RGB).
        target_spectrum (Optional[np.ndarray]). A target spectrum to compute rotational average to compare against.
        normalize_rmse (bool): Whether to normalize the RMSE

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            - Correlation coefficients (np.ndarray) between the rotational averages
              of the input images and the target spectrum.
            - Root mean square errors (np.ndarray) for the same comparison.
    """

    x_size, y_size = images[0].shape[:2]
    n_channels = 1 if images[0].ndim == 2 else 3

    # Compute spectra and mean spectrum if required
    magnitudes, phases = get_images_spectra(images=images)
    if target_spectrum is None:
        target_spectrum = im3D(np.zeros(images[0].shape))
        for idx, mag in enumerate(magnitudes):
            target_spectrum += mag
        target_spectrum /= len(magnitudes)
    target_spectrum = im3D(target_spectrum)

    # Returns the frequencies of the image, bins range from -0.5f to 0.5f (0.5f is the Nyquist frequency) 1/y_size is the distance between each pixel in the image
    r_int = get_radius_grid(x_size=x_size, y_size=y_size)
    target_rot_avg = []
    initial_rot_avg = []
    for idx, image in enumerate(images):
        magnitude = im3D(magnitudes[idx])
        tra = []
        ira = []
        for channel in range(n_channels):
            fft_image = magnitude[:, :, channel]

            # Rotational averages (target vs source) as MEANS over annuli
            ira.append(rotational_avg(spectrum=fft_image, radius=r_int))
            if idx == 0:
                tra.append(rotational_avg(spectrum=target_spectrum[..., channel], radius=r_int))
        if idx == 0:
            target_rot_avg = np.stack(tra).T
        initial_rot_avg.append(np.stack(ira).T)

    # Compute metrics
    N = len(initial_rot_avg)
    corr, rms = np.zeros((N,)), np.zeros((N,))
    for idx, ira in enumerate(initial_rot_avg):
        corr[idx] = np.corrcoef(ira.ravel(), target_rot_avg.ravel())[0, 1]
        if normalize_rmse:
            rms[idx] = normalized_rmse(ira.ravel(), target_rot_avg.ravel(), mode='actual range')
        else:
            rms[idx] = compute_rmse(ira.ravel(), target_rot_avg.ravel())
    return corr, rms


def spec_match_validation(images: ImageListIO, target_spectrum: Optional[np.ndarray] = None, normalize_rmse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates spectral matching of input images by comparing the spectra of each
    image with a target spectrum. The target spectrum is computed as the average
    magnitude spectrum of all input images. Computes both the correlation and root
    mean square error (RMSE) between the magnitude spectra of individual images
    and the target spectrum.

    Args:
        images: List or array of images for which spectral matching needs to be
            validated. Each image should have the same shape.
        target_spectrum (Optional[np.ndarray]). A target spectrum to compare against.
        normalize_rmse (Optional[bool]): Whether to normalize the RMSE

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
        first array contains the correlation coefficients between the magnitude
        spectra of individual images and the target spectrum. The second array
        contains the root mean square errors (RMSE) for the same comparison.
    """
    magnitudes, phases = get_images_spectra(images=images)

    n_channels = 1 if images[0].ndim == 2 else 3
    compute_target_spectrum = target_spectrum is None
    if not compute_target_spectrum:
        n_channels_ts = target_spectrum.shape[2] if target_spectrum.ndim == 3 else 1
        if n_channels_ts != n_channels:
            raise ValueError(f"Target spectrum has {n_channels_ts} channels but images have {n_channels} channels")

    target_spectrum = im3D(np.zeros(images[0].shape)) if compute_target_spectrum else im3D(target_spectrum)
    if compute_target_spectrum:
        for idx, mag in enumerate(magnitudes):
            target_spectrum += mag
        target_spectrum /= len(magnitudes)
        target_spectrum = im3D(target_spectrum)

    # Compute metric
    N = len(magnitudes)
    corr, rms = np.zeros((N,)), np.zeros((N,))
    for idx, a_mag in enumerate(magnitudes):
        a_mag = im3D(a_mag)
        corr[idx] = np.corrcoef(a_mag.ravel(), target_spectrum.ravel())[0, 1]
        if normalize_rmse:
            rms[idx] = normalized_rmse(a_mag.ravel(), target_spectrum.ravel(), mode='actual range')
        else:
            rms[idx] = compute_rmse(a_mag.ravel(), target_spectrum.ravel())
    return corr, rms


def compute_rmse(image1: np.ndarray, image2: np.ndarray) -> float:
    """ Compute the root-mean-square error between two images. """
    return np.sqrt(np.mean((image1 - image2) ** 2))


def normalized_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: Literal["assume [0, 1]", "actual range", "expected range"] = "actual range",
    expected_range: Optional[float] = None,
    eps: float = 1e-12,
    verbose: bool = False,
) -> float:
    """Compute RMS error guaranteed to lie in [0, 1].

    Args:
        y_true: Reference array.
        y_pred: Predicted or reconstructed array (same shape as y_true).
        mode:
            - "assume [0, 1]": assumes both arrays are already scaled to [0, 1];
              RMSE is directly bounded to [0, 1].
            - "actual range": divides by the actual dynamic range of y_true
              (max(y_true) - min(y_true)).
            - "expected range": divides by a user-specified `expected_range`
              (e.g. 255 for 8-bit data or 1.0 for normalized floats).
        expected_range: Known dynamic range when using "expected range".
        eps: Small constant to avoid divide-by-zero.
        verbose: Out-of-range value warnings if True

    Returns:
        float: Normalized RMSE in [0, 1].

    Raises:
        ValueError: If shapes differ or `expected_range` is missing when required.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match.")

    rmse = compute_rmse(y_true, y_pred)
    if mode == "assume [0, 1]":
        # Data already in [0,1], so RMSE is naturally bounded.
        norm_rmse = rmse

    if mode == "actual range":
        actual_range = float(np.max([np.max(y_true), np.max(y_pred)]) - np.min([np.min(y_true), np.min(y_pred)]))
        norm_rmse = rmse / max(actual_range, eps)

    if mode == "expected range":
        if expected_range is None:
            raise ValueError("`expected_range` must be provided for mode='expected range'.")
        norm_rmse = rmse / max(expected_range, eps)

    if verbose and not (0.0 <= norm_rmse <= 1.0):
        warnings.warn(
            f"Normalized RMSE ({norm_rmse:.4f}) is outside [0, 1]. "
            f"This suggests inconsistent range assumptions.",
            RuntimeWarning,
            stacklevel=2,
        )

    return norm_rmse


def get_images_spectra(images: ImageListIO, magnitudes: Optional[ImageListIO] = None, phases: Optional[ImageListIO] = None, rescale: bool = True) -> Tuple[ImageListIO, ImageListIO]:
    """
    Get spectrum over list of images
    Args:
        images (ImageListIO): List of images.
        magnitudes (Optional[ImageListIO]): If provided, inserts new magnitudes into this list.
        phases (Optional[ImageListIO]): If provided, inserts new phases into this list.
        rescale (bool): Determines if input is stretched to [0, 1] range.

    Returns:
        magnitudes, phases (Tuple[ImageListIO, ImageListIO])

    """
    n_images = len(images)
    x_size, y_size = images[0].shape[:2]
    n_channels = 3 if len(images[0].shape) == 3 else 1
    phases = [None] * n_images if phases is None else phases
    magnitudes = [None] * n_images if magnitudes is None else magnitudes
    for idx, image in enumerate(images):
        if images.drange == (0, 255):
            image = np.float64(image)/255
        magnitudes[idx], phases[idx] = image_spectrum(image, rescale=rescale)
    return magnitudes, phases


def rescale_image(image: np.ndarray, target_min: Optional[float] = 0, target_max: Optional[float] = 1) -> np.ndarray:
    """
    Rescale an image to the range: [target_min, target_max]

    Args:
        image (np.ndarray): Input image
        target_min (float): Target minimum value in the output image.
        target_max (float): Target maximum value in the output image.

    Returns:
        (np.ndarray): Rescaled image
    """

    # Convert to float
    image = image.astype(float)

    # Rescale to the specified range: default is [0, 1]
    image -= np.min(image)
    image /= np.max(image)
    image *= (target_max - target_min)
    image += target_min
    return image


def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """Load all supported image formats from a folder as NumPy arrays.

    This function searches a given folder for image files with supported extensions
    (e.g., PNG, JPEG, TIFF, BMP, WEBP, GIF) and loads them into memory as NumPy arrays.
    Non-image files are ignored. Any unreadable images are skipped with an error message.

    Args:
        folder_path: Path to the folder containing images.

    Returns:
        A list of NumPy arrays representing the loaded images. Returns an empty list
        if the folder does not exist or no valid images are found.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        console_log(f"Folder not found: {folder}", color=Bcolors.FAIL)
        return []

    supported_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in supported_exts]

    if not image_files:
        console_log(f"No supported image files found in: {folder}", color=Bcolors.WARNING)
        return []

    arrays: List[np.ndarray] = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                arrays.append(np.array(img))
        except Exception as e:
            console_log(f"Failed to load {img_path.name}: {e}", color=Bcolors.FAIL)

    console_log(f"Loaded {len(arrays)} image(s) from {folder}", color=Bcolors.OKGREEN)
    return arrays


def load_np_array(path_str: Optional[str]) -> Optional[np.ndarray]:
    """
    Loads a NumPy array from a specified file path. If the file path is not provided,
    does not exist, or is not a supported file type (only `.npy` files are supported),
    an appropriate message is logged and `None` is returned.

    Args:
        path_str (Optional[str]): The file path to the `.npy` file. If not provided
            or invalid, the function returns `None`.

    Returns:
        Optional[np.ndarray]: Loaded NumPy array if the file exists and is
            successfully loaded; `None` otherwise.
    """
    if not path_str:
        return None
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        console_log(f"✗ File not found: {p}", indent_level=1, color=Bcolors.FAIL)
        return None
    if p.suffix.lower() == ".npy":
        return np.load(p, allow_pickle=False)
    console_log(f"✗ Unsupported file type: {p.suffix} (only .npy supported)", indent_level=1, color=Bcolors.FAIL)
    return None


def rescale_images255(images: ImageListIO, rescaling_option: Literal[0, 1, 2, 3] = 2, legacy_mode: bool = False) -> ImageListIO:
    """
    Rescales the values of images so that they fall between 0 and 255. There are 3 options:
        1) Each image has its own min and max (no rescaling)
        2) Each image is rescaled so that the absolute max and min values obtained across all images are between 0 and 255
        3) Each image is rescaled so that the average max and min values obtained across all images are between 0 and 255

        Args:
            images : list of images
            rescaling_option: (optional) : Determines the type of rescaling.
                0 : No rescaling
                1 : Rescaling each image so that it stretches to [0, 1]
                2 : Rescaling absolute max/min (Default)
                3 : Rescaling average max/min
            legacy_mode (bool): If true, only the absolute max/min values are rescaled.
            legacy_mode (bool): If true, only the absolute max/min values are rescaled.

        Returns :
            A list of rescaled images

        Notes:
            Warning: Always returns a [0, 255] image, no matter the range of the input images.
    """

    if rescaling_option not in [0, 1, 2, 3]:
        raise ValueError(f'The rescaling option must be either [0, 1, 2, 3], now rescaling is : {rescaling_option}')

    n_images = len(images)
    minimum_values = np.zeros((n_images,))
    maximum_values = np.zeros((n_images,))
    for idx, image in enumerate(images):
        minimum_values[idx], maximum_values[idx] = np.min(image), np.max(image)

    mn, mx = None, None
    if rescaling_option == 2:
        mn, mx = np.min(minimum_values), np.max(maximum_values)
    elif rescaling_option == 3:
        mn, mx = np.mean(minimum_values), np.mean(maximum_values)

    if rescaling_option:
        for idx, image in enumerate(images):
            new_image = image.copy().astype(float)
            if rescaling_option == 1:
                new_image = stretch(new_image) * 255
            else:
                new_image = (new_image - mn)/(mx - mn) * 255
            images[idx] = MatlabOperators.uint8(new_image) if legacy_mode else new_image

    return images


def uint8_plus(image: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Apply clipping, banker's rounding and uint8 transformations.

    Args:
        image (np.ndarray): Image to be converted. Assume that the input image is already in the proper range [0, 255].
        verbose (bool): Warn if clipping needed.

    Returns:
        image (np.ndarray)

    """
    if verbose:
        mn, mx = image.min(), image.max()
        if mn < 0 or mx > 255:
            print(f'{Bcolors.WARNING}Out of range values: Actual range [{mn}, {mx}] outside of the admitted range [0, 255]\nValues will be clipped.{Bcolors.ENDC}')

    return np.rint(np.clip(image, 0, 255)).astype('uint8')


def apply_median_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a median blur to image.

    Parameters:
        image (np.ndarray): Input image (2D grayscale or 3D multi-channel).
        kernel_size (int): Size of the square kernel (must be an odd integer).

    Returns:
        np.ndarray: Blurred image

    Tested by Nicolas D.R.: Should provide exact same results on np.uint8 as cv2.medianBlur.
    """
    if kernel_size % 2 == 0:
        raise ValueError("median_blur_force must be an odd integer.")
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be a 2D or 3D array.")

    original_ndim = image.ndim
    image_3d = im3D(image)
    pad = kernel_size // 2

    # Pad spatial dimensions only
    padded = np.pad(image_3d, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='edge')

    # Sliding window over H and W only, channels preserved
    windows = sliding_window_view(padded, (kernel_size, kernel_size, 1))
    # Result shape: (H, W, C, k, k, 1)

    # Move (k, k) to the end, then flatten and compute median
    windows = windows.squeeze(-1)  # shape (H, W, C, k, k)
    k = kernel_size
    windows = windows.reshape(*windows.shape[:3], k * k)  # (H, W, C, k*k)

    # Apply filter and restore original dimensions
    blurred = np.median(windows, axis=-1).astype(image.dtype)
    blurred = blurred[:, :, 0].squeeze() if original_ndim == 2 else blurred

    return blurred


def hist2list(hist: np.ndarray) -> np.ndarray:
    """
    Converts a luminance histogram of an image into a sorted list of luminance values.

    Args:
        hist (np.ndarray): Luminance histogram (counts of each intensity) for each channel.

    Returns:
        np.ndarray: Sorted list of luminance values for each channel. It has the same number
            of channels as hist.There is one list per channel and its size is equal to  np.sum(hist[:, channel]).
    """
    if hist.shape[1]>1 and not np.unique(np.sum(hist, axis=1)).shape[0] != 1:
        raise ValueError(f"All 3 channels of hist must have the same sum")
    final_lists = np.zeros((int(np.sum(hist[:, 0])), hist.shape[1]))
    for channel in range(hist.shape[1]):
        final_lists[:, channel] = np.stack(list(chain.from_iterable([val] * cnt for val, cnt in enumerate(hist[:, channel])))).astype(int)
    return final_lists


def im3D(image: np.ndarray):
    """ Forces a third dimension on grayscale image.

    Args:
        image (np.ndarray): 2D or 3D image.

    Returns:
        image (np.ndarray) with 3D: grayscale (H, W) -> (H, W, 1); RGB (H, W, 3) stay the same.

    """
    image = np.asarray(image)
    return np.stack((image,), axis=-1) if image.ndim != 3 else image


def imhist(image: np.ndarray, mask: Optional[np.ndarray] = None, n_bins: int = 256, normalized: bool = False) -> np.ndarray:
    """Computes the histogram of the image. If RGB image, it provides one hist per channel.

        Args:
            image (np.ndarray): Image (ndarray).
            mask (np.ndarray): If a boolean mask is provided, computes the histogram within the mask (ndarray).
            n_bins: Number of bins for the histogram (default is 256).
            normalized (bool): If yes, the output hist will sum to 1 (default = False)
        Returns:
            counts (np.ndarray): Histogram counts for each channel.
    """

    # Force a third dimension to image in case it only has two
    image = im3D(image)

    # If no mask provided, make a blank mask with all True
    mask = np.ones(image.shape).astype(bool) if mask is None else im3D(mask.astype(bool))
    mask = np.repeat(mask, image.shape[2], axis=2) if mask.shape[2] < image.shape[2] else mask
    n_channels = image.shape[-1]
    count = np.zeros((n_bins, n_channels))
    for channel in range(n_channels):
        count[:, channel], _ = np.histogram(image[:, :, channel][mask[:, :, channel]], bins=n_bins, range=(0, n_bins))

        if normalized:
            count[:, channel] = count[:, channel] / count[:, channel].sum()

    return count


def avg_hist(images: ImageListIO, binary_masks: List[np.ndarray], normalized: bool = True, n_bins: int = 256, output_list_hist: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """Computes the average histogram of a set of images.

    Args:
        images (ImageListIO): A list of images
        binary_masks (List[np.ndarray]): A list of binary mask.
        normalized (bool): Indicate of the result should be normalize to sum to 1.
        n_bins (int): Number of levels in the image (uint8 = 256)
        output_list_hist (bool): If True, outputs a list of histogram corresponding to each image in images (along with the average)

    Returns:
        average (Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]): Average histogram counts for each channel.

    """
    def normalize_hist(a_hist):
        a_hist = np.float64(a_hist)
        return a_hist / (a_hist.sum(axis=0, keepdims=True) + 1e-12)

    n_channels = 1 if images.n_dims == 2 else 3
    if len(binary_masks) != images.n_images:
        raise ValueError('Length of binary_masks should be equal to length of images')

    hist_sum = np.zeros((n_bins, n_channels))
    hist_list = []
    for idx, im in enumerate(images):
        hist_list.append(imhist(im, binary_masks[idx], n_bins=n_bins))
        hist_sum += hist_list[-1]
        if normalized:
            hist_list[-1] = normalize_hist(hist_list[-1])

    # Average of the pixels in the bins
    average = hist_sum / len(images)
    if normalized:
        average = normalize_hist(average)

    if output_list_hist:
        return average, hist_list
    else:
        return average

