# TODO: refactor class and function names; refactor docstrings to google style; get rid of cv2
# TODO: Before V1 commit: Remove all revision comments (e.g. see round in MatlabOperators)
# TODO: Before V1 commit: Remove debug points
# TODO: Optimization: Check image type and use np.fft.rfft2 for faster computations.

# External package imports
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Any, Optional, Tuple, Union, NewType, List, Iterator, Callable, Literal
from PIL import Image
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re

# Local package imports

# Type definition
ImageListType = Union[str, Path, List[Union[str, Path]], List[np.ndarray]]
RGB2GRAY_WEIGHTS = {
    'equal': [1 / 3, 1 / 3, 1 / 3],
    '709': [0.2125, 0.7154, 0.0721],
    '601': [0.299, 0.587, 0.114],
    '2020': [0.2627, 0.6780, 0.0593],
}
for k, v in RGB2GRAY_WEIGHTS.items():
    RGB2GRAY_WEIGHTS[k] /= np.sum(v)
int2key_mapping = dict(zip(range(len(RGB2GRAY_WEIGHTS)), RGB2GRAY_WEIGHTS.keys()))
RGB2GRAY_WEIGHTS['int2key'] = int2key_mapping
RGB2GRAY_WEIGHTS['key2int'] = dict(zip(RGB2GRAY_WEIGHTS['int2key'].values(), RGB2GRAY_WEIGHTS['int2key'].keys()))


class Bcolors:
    HEADER = '\033[95m'  # Processing steps
    OKBLUE = '\033[94m'  # Processing values
    OKCYAN = '\033[96m'  # Internal notes
    OKGREEN = '\033[92m'  # Ok values
    WARNING = '\033[93m'
    FAIL = '\033[91m'  # Problematic values
    ENDC = '\033[0m'
    BOLD = '\033[1m'  # Iteration
    UNDERLINE = '\033[4m'
    SECTION = '\033[4m\033[1m'  # Image loop


class MatlabOperators:

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
        if image.ndim == 3:
            return np.dot(image.astype(np.float64), RGB2GRAY_WEIGHTS['601'])
        else:
            return image


def imhist_plot(
    img: np.ndarray,
    bins: int = 256,
    figsize=(8, 6),
    dpi=100,
    normalize: bool = False,
    title: Optional[str] = None,
):
    """
    Display an image on top and a compact horizontal histogram underneath.
    A grayscale gradient bar (0..255) is placed *flush* under the histogram x-axis.

    Args:
        img: np.ndarray, shape (H, W) or (H, W, C). Supports uint8, float in [0,1] or [0,255].
        bins: number of histogram bins (default 256).
        figsize, dpi: matplotlib figure size and dpi.
        normalize: if True, plot histograms as densities (area=1). Otherwise raw counts.
        title: optional string title.

    Returns:
        (fig, (ax_img, ax_bar, ax_hist))
    """
    # ---- normalize image to uint8, ignore alpha if present ----
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[..., :3]  # drop alpha

    if np.issubdtype(arr.dtype, np.floating):
        a, b = float(np.nanmin(arr)), float(np.nanmax(arr))
        if b <= 1.0:  # assume [0,1]
            arr = np.clip(arr, 0, 1) * 255.0
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    elif not np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(arr.astype(np.float64), 0, 255)
        arr = np.rint(arr).astype(np.uint8)

    # handle grayscale vs RGB for display
    is_rgb = (arr.ndim == 3 and arr.shape[2] == 3)
    if arr.ndim == 2:
        arr_rgb_for_show = np.stack([arr]*3, axis=-1)   # show as gray-to-RGB
    else:
        arr_rgb_for_show = arr

    # ---- histograms ----
    edges = np.linspace(0, 256, bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if is_rgb:
        Hr, _ = np.histogram(arr[..., 0].ravel(), bins=edges, density=normalize)
        Hg, _ = np.histogram(arr[..., 1].ravel(), bins=edges, density=normalize)
        Hb, _ = np.histogram(arr[..., 2].ravel(), bins=edges, density=normalize)
        Hmax = max(Hr.max(), Hg.max(), Hb.max()) if not normalize else 1.0
    else:
        Hy, _ = np.histogram(arr.ravel(), bins=edges, density=normalize)
        Hmax = Hy.max() if not normalize else 1.0

    # ---- figure & axes (make histogram width match image width) ----
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.5, 1.4], hspace=0.12)

    ax_img  = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # Image on top
    ax_img.imshow(arr_rgb_for_show, interpolation='nearest')
    ax_img.axis('off')
    if title:
        ax_img.set_title(title, fontsize=11)

    # Ensure histogram axes have EXACT same left/right as image axes
    fig.canvas.draw()  # compute positions
    img_pos  = ax_img.get_position()
    hist_pos = ax_hist.get_position()
    ax_hist.set_position([img_pos.x0, hist_pos.y0, img_pos.width, hist_pos.height])

    # Plot histogram(s)
    if is_rgb:
        ax_hist.plot(centers, Hr, lw=1.5, color='red',   label='R')
        ax_hist.plot(centers, Hg, lw=1.5, color='green', label='G')
        ax_hist.plot(centers, Hb, lw=1.5, color='blue',  label='B')
        ax_hist.legend(frameon=False, fontsize=9, loc='upper right')
    else:
        ax_hist.plot(centers, Hy, lw=1.5, color='black', label='Y')
        ax_hist.legend(frameon=False, fontsize=9, loc='upper right')

    ax_hist.set_xlim(0, 255)
    ax_hist.set_ylim(0, Hmax * 1.05 if Hmax > 0 else 1)
    ax_hist.set_yticks([])
    ax_hist.set_xticks([])  # no numbers; we'll show a gradient bar instead
    for spine in ("top", "right"):
        ax_hist.spines[spine].set_visible(False)

    # ---- grayscale gradient bar FLUSH under histogram x-axis ----
    divider = make_axes_locatable(ax_hist)
    divider = make_axes_locatable(ax_hist)
    ax_bar = divider.append_axes("bottom", size="5%", pad=0.0)  # was "6mm"
    gradient = np.linspace(0, 1, 256, dtype=np.float64).reshape(1, -1)
    ax_bar.imshow(gradient, cmap='gray', aspect='auto', extent=[0, 255, 0, 1])
    ax_bar.set_xlim(ax_hist.get_xlim())
    ax_bar.set_xticks([])   # no numbers
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    return fig, (ax_img, ax_bar, ax_hist)


def sf_plot(im: np.ndarray, qplot: bool = True) -> np.ndarray:
    """
    Rotational average of the Fourier energy spectrum.

    Parameters
    ----------
    im : np.ndarray
        Image array of shape (H, W) or (H, W, 3). Can be uint8 or float.
        RGB is converted to luminance (ITU-R BT.601).
    qplot : bool, default True
        If True, plot loglog spectrum (cycles/image vs energy).

    Returns
    -------
    avg : np.ndarray
        1D array of rotationally averaged energy for integer radii
        1..floor(min(H, W)/2), matching the MATLAB implementation.
    """

    # --- to grayscale float64 ---
    arr = np.asarray(im)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # MATLAB rgb2gray-like (double precision)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        if np.issubdtype(arr.dtype, np.integer):
            r = r.astype(np.float64); g = g.astype(np.float64); b = b.astype(np.float64)
        gray = rgb2gray(arr, conversion_type='rec601')
    else:
        gray = arr.astype(np.float64, copy=False)

    xs, ys = gray.shape  # xs = rows (y), ys = cols (x)

    # --- Fourier energy (fftshifted) ---
    fftim = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2

    # --- frequency grids replicating MATLAB logic ---
    def freq_axis(n: int) -> np.ndarray:
        # MATLAB:
        # even n:   -n/2 : n/2-1
        # odd  n:   -n/2 : n/2-1  (with halves → -2.5,-1.5,...,+1.5 for n=5)
        if n % 2 == 0:
            return np.arange(-n//2, n//2, dtype=np.float64)
        else:
            # center at half-steps to avoid 0; e.g., n=5 -> -2.5..+1.5
            half = n // 2
            return np.arange(-(half + 0.5), half + 0.5, 1.0, dtype=np.float64)

    f2 = freq_axis(xs)  # rows
    f1 = freq_axis(ys)  # cols
    XX, YY = np.meshgrid(f1, f2)  # shape (xs, ys)

    # --- polar radius, MATLAB rounding rule ---
    r = np.hypot(XX, YY)
    if (xs % 2 == 1) or (ys % 2 == 1):
        r = np.rint(r) - 1.0
    else:
        r = np.rint(r)

    # Non-negative integer bin indices
    r = np.clip(r, 0, None).astype(np.int64)

    # --- accumarray equivalent: mean energy per radius ---
    flat_r = r.ravel()
    flat_e = fftim.ravel()
    sums = np.bincount(flat_r, weights=flat_e)
    counts = np.bincount(flat_r)
    counts[counts == 0] = 1  # guard against divide-by-zero
    avg_full = sums / counts

    # Match MATLAB: avg = avg(2:floor(min(xs,ys)/2)+1)
    R = int(np.floor(min(xs, ys) / 2.0))
    radii = np.arange(1, R + 1)
    avg = avg_full[1:R + 1]

    if qplot:
        plt.figure()
        plt.loglog(radii, avg)
        plt.xlabel('Spatial frequency (cycles/image)')
        plt.ylabel('Energy')
        plt.tight_layout()

    return avg


def spectrum_plot(spectrum: np.ndarray,
                  cmap: str = "gray",
                  log: bool = True,
                  gamma: float = 1.0,
                  with_colorbar: bool = True):
    """Display a Fourier magnitude spectrum with optional log and gamma scaling."""
    spec = np.abs(spectrum).astype(np.float64)

    # log scaling
    if log:
        spec = np.log1p(spec)

    # stretch to [0,1]
    spec = (spec - spec.min()) / (spec.max() - spec.min())

    # gamma correction
    if gamma != 1.0:
        spec = spec ** gamma

    plt.imshow(spec, cmap=cmap)
    if with_colorbar:
        plt.colorbar()
    plt.show()


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


def pixel_order(image: np.ndarray) -> Tuple[np.ndarray, Union[float, list]]:
    """Assign strict ordering to monochromatic or multispectral image pixels.

    For each channel, builds a 6-D feature vector per pixel:
      F1 = raw pixel value
      F2 = 3×3 cross mean (not separable)
      F3 = 3×3 box mean (separable)
      F4 = 5×5 ring-ish mean (not separable)
      F5 = 5×5 box w/o corners (not separable)
      F6 = 5×5 full box mean (separable)

    Then sorts pixels lexicographically by (F1, F2, F3, F4, F5, F6).
    Returns per-channel rank maps and order accuracy (OA).

    Args:
        image (np.ndarray): Grayscale (H, W) or color (H, W, C) image.

    Returns:
        im_sort (np.ndarray): (H, W, C) rank maps (or (H, W, 1) for grayscale).
        OA: Order accuracy in [0, 1]; list per channel for C>1, else float.
    """
    image = im3D(image)
    M, N, P = image.shape

    # --- Define filters ---
    # F2: 3×3 cross (not separable)
    F2 = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.float64) / 5.0

    # F3: 3×3 box (separable: g3 ⊗ g3)
    g3 = np.array([1.0, 1.0, 1.0], dtype=np.float64) / 3.0

    # F4: 5×5 ring-ish (not separable)
    F4 = np.ones((5, 5), dtype=np.float64) / 13.0
    F4[[0, 0, 1, 1, 1, 3, 3, 4, 4, 4], [0, 1, 0, 1, 4, 0, 4, 1, 3, 4]] = 0.0

    # F5: 5×5 box without corners (not separable)
    F5 = np.ones((5, 5), dtype=np.float64) / 21.0
    F5[[0, 0, 4, 4], [0, 4, 0, 4]] = 0.0

    # F6: 5×5 full box (separable: g5 ⊗ g5)
    g6 = np.ones(5, dtype=np.float64) / 5.0

    im_sort = []
    OA = []

    for c in range(P):
        ch = image[:, :, c].astype(np.float64, copy=False)

        # Build feature responses FR[..., 0..5]
        FR = np.zeros((M, N, 6), dtype=np.float64)

        # F1: identity
        FR[:, :, 0] = ch

        # F2: 3×3 cross (2D)
        FR[:, :, 1] = convolve_2d(ch, F2)  # 2D kernel

        # F3: 3×3 box (separable)
        FR[:, :, 2] = convolve_2d(ch, g3)  # 1D -> separable path

        # F4: 5×5 ring-ish (2D)
        FR[:, :, 3] = convolve_2d(ch, F4)  # 2D kernel

        # F5: 5×5 box without corners (2D)
        FR[:, :, 4] = convolve_2d(ch, F5)  # 2D kernel

        # F6: 5×5 full box (separable)
        FR[:, :, 5] = convolve_2d(ch, g6)  # 1D -> separable path

        # Rearrange the filter responses
        FR = FR.reshape(M * N, 6)

        # Number of unique filter responses and ordering accuracy
        n_unique = np.unique(FR, axis=0).shape[0]
        OA.append(n_unique / (M * N))

        # Sort responses lexicographically
        # [:, ::-1] because np.lexsort applies sort keys from last to first (right to left).
        idx_pos = np.lexsort(FR[:, ::-1].T)

        # Rearrange indices according to pixel position
        idx_rank = np.argsort(idx_pos).reshape(M, N)
        im_sort.append(idx_rank)

    if P == 1:
        OA = OA[0]

    return np.stack(im_sort, axis=-1), OA


def exact_histogram_with_noise(image: np.ndarray, target_hist: np.ndarray, binary_mask: Optional[np.ndarray] = None, noise_level: float = 0.1, n_bins: int = 256) -> np.ndarray:
    """Exact histogram specification by rank allocation (mask-aware, per channel).

    This implements a discrete, “exact” histogram specification: masked pixels are
    ranked (with small amount of noise for tie-breaking), then assigned to output levels so
    that the counts per level match `target_hist` as closely as possible
    (exact up to rounding and mask size). Unmasked pixels are left unchanged.

    Args:
        image (np.ndarray): Input image, 2D (H,W) or 3D (H,W,C). Any numeric dtype.
            If Float, values are used only for ranking; the output levels are in
            the integer range [0, n_bins-1].
        target_hist (np.ndarray): Target histogram counts or weights with shape
            (n_bins, C). If weights, they are normalized internally to the number
            of masked pixels in each channel.
        binary_mask (np.ndarray): Boolean mask, shape (H,W) or (H,W,C). True indicates
            pixels to be histogram-specified. If 2D, it is applied to all channels.
        noise_level (float): Level of noise used only to break ties
            in ranking. Set to a small value like 1e-3 (relative to input scale).
            Use 0 to disable noise (less robust when many ties).
        n_bins (int): Number of discrete output levels (e.g., 256, 65536).

    Returns:
        np.ndarray: Output image with masked pixels reassigned to match the target
        histogram, dtype chosen from {uint8, uint16, uint32} according to
        `n_bins`. Unmasked pixels keep their original values (cast to output dtype).


    Reference: Coltuc, Dinu; Bolon, Philippe; Chassery, Jean-Marc. Exact Histogram Specification. IEEE Transactions on Image Processing, Vol. 15, No. 5, May 2006, pp. 1143-1152. doi:10.1109/TIP.2005.864170
    """
    # Ensure 3D (H,W,C)
    im = im3D(image)  # your helper: returns (H,W,C)
    H, W, C = im.shape
    mask = im3D(binary_mask) if binary_mask is not None else np.zeros_like(im, dtype=bool)
    mask = np.broadcast_to(mask, (H, W, C)) if mask.shape[2] == 1 and C > 1 else mask

    # Validate target histogram shape
    if target_hist.shape[0] != n_bins:
        raise ValueError(f"target_hist must have shape (n_bins, C) with n_bins={n_bins}; "
                         f"got {target_hist.shape[0]} bins.")
    if target_hist.shape[1] != C:
        raise ValueError(f"target_hist has {target_hist.shape[1]} channels, but image has {C}.")

    n_bits = int(np.log2(n_bins))
    if n_bins not in (256, 65536, 4294967296) and 2 ** n_bits != n_bins:
        raise ValueError(f"n_bins must be a power of two; got {n_bins}.")
    out_dtype = f'uint{n_bits}'
    out = np.empty((H, W, C), dtype=out_dtype)

    # --- Per-channel exact allocation ---
    for c in range(C):
        ch_mask = mask[:, :, c].astype(bool)
        idx = np.flatnonzero(ch_mask.ravel())
        if idx.size == 0:
            raise ValueError(f"Mask for channel {c} has no True elements.")

        th = np.maximum(target_hist[:, c].astype(np.float64), 0.0)
        s = th.sum()
        if s <= 0.0:
            raise ValueError(f"Target histogram for channel {c} sums to zero.")

        # Values used for ranking (float); add small jitter to break ties
        x = im[..., c].ravel()[idx].astype(np.float64)
        x = x + np.random.uniform(-noise_level, noise_level, size=x.shape).astype(np.float64)

        order = np.argsort(x, kind="mergesort")  # stable rank order

        # Desired integer counts per level (unbiased rounding)
        desired = th / s * idx.size
        base = np.floor(desired).astype(np.int64)
        rem = idx.size - int(base.sum())
        if rem != 0:
            frac = desired - base
            add_idx = np.argsort(-frac)[:abs(rem)]
            base[add_idx] += np.sign(rem)

        # Assign contiguous blocks of ranks to output levels 0..n_bins-1
        assign = np.empty(idx.size, dtype=out_dtype)
        start = 0
        for level, k in enumerate(base):
            if k > 0:
                assign[order[start:start + k]] = np.asarray(level, dtype=out_dtype)
                start += k
        if start != idx.size:  # should not happen; strict fill
            raise RuntimeError(f"Rounding error while assigning ranks in channel {c}.")

        # Write back masked pixels; copy-through others (cast)
        ch_out = np.asarray(im[..., c], dtype=out_dtype).ravel()
        ch_out[idx] = assign
        out[..., c] = ch_out.reshape(H, W)

    return out.squeeze()


def exact_histogram(image: np.ndarray, target_hist: np.ndarray, binary_mask: np.ndarray = None, n_bins: Optional[int] = None, verbose: bool = True) -> Tuple[np.ndarray, List]:
    """
    Specify exact image histogram.

    Args:
        image (np.ndarray): Input image (8-bit or 16-bit grayscale or RGB).
        target_hist (np.ndarray): Specified histogram.
        binary_mask (np.ndarray): Binary mask to only adjust pixel intensities in the foreground (optional).
        n_bins (int): If provided, will be used to set L value. This is convenient when using float images.
        verbose (bool): Log information

    Returns:
        Tuple[np.ndarray, Union[float, list]]:
            - im_out (np.ndarray): Processed image with specified histogram.
            - OA (list): Order accuracy indicating fraction of unique filter response combinations.

    References:
        1. Coltuc D. and Bolon P., 1999, "Strict ordering on discrete images
        and applications"
        2. Coltuc D., Bolon P. and Chassery J-M., 2006, "Exact histogram
        specification", IEEE Transactions on Image Processing
        15(5):1143-1152

    This code is a Python implementation of Anton Semechko's (asemechk@uoguelph.ca) exact_histogram in MATLAB
    """

    # Maximum number of gray levels and image dimensions
    L = n_bins if n_bins is not None else None
    if L is None and np.issubdtype(image.dtype, np.integer):
        L = 2 ** np.iinfo(image.dtype).bits
    if L is None:
        raise ValueError("L, the expected number of values per channel, must be specified!")

    image = im3D(image)  # force a third dimension on image
    x_size, y_size, n_channels = image.shape

    # Verify input format
    if image.dtype not in (np.uint8, np.uint16) and not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Input image must be 8- or 16-bit or float")
    if len(target_hist) != L:
        raise ValueError("Number of histogram bins must match maximum number of gray levels.")
    if target_hist.ndim != 2 or target_hist.shape[1] != n_channels:
        raise ValueError("Target histogram (target_hist) should have the same number of channels as the image.")
    if binary_mask is not None:
        binary_mask = im3D(binary_mask)  # force a third dimension on image
        if not image.shape == binary_mask.shape:
            raise ValueError(f"binary_mask shape ({binary_mask.shape}) should be equal to image shape ({image.shape})")
        if np.sum(binary_mask) < (50 * n_channels):
            raise ValueError("Too few foreground pixels in the binary mask.")
    else:
        binary_mask = np.ones(image.shape, dtype=bool)
    if n_channels not in [1, 3]:
        raise ValueError("Input image must have 1 or 3 channels.")

    # Assign strict order to pixels
    # print(f"{Bcolors.HEADER}Assigning strict order to pixels{Bcolors.ENDC}") if verbose else None
    im_sort, OA = pixel_order(image)

    # Process each channel separately
    # print(f"{Bcolors.HEADER}Main exact_histogram loop{Bcolors.ENDC}") if verbose else None
    im_out = image.copy()
    for channel in range(n_channels):
        # Work only on the masked (foreground) pixels
        foreground_indices = binary_mask[:, :, channel]
        Ntotal = np.sum(foreground_indices)

        # Get the pixel order values for the masked region
        pix_ord = im_sort[:, :, channel][foreground_indices]

        # Sort pixel order and get the sorted indices
        sorted_indices = np.argsort(pix_ord)

        # Adjust the specified histogram to match the number of pixels in the mask
        new_target_hist = (Ntotal * target_hist[:, channel] / np.sum(target_hist[:, channel])).astype(int)
        residuals = Ntotal - np.sum(new_target_hist)

        # Redistribute the residuals to ensure total counts match
        sorted_residuals = np.argsort(-np.mod(Ntotal * target_hist[:, channel] / np.sum(target_hist[:, channel]), 1))
        new_target_hist[sorted_residuals[:residuals]] += 1

        # Create intensity values based on the adjusted histogram
        Hraw = np.repeat(np.arange(L), new_target_hist)

        # Reorder Hraw according to the sorted pixel positions
        Hraw_sorted = np.zeros_like(Hraw)
        Hraw_sorted[sorted_indices] = Hraw

        # Assign the sorted intensity values back to the output image
        im_out[:, :, channel][foreground_indices] = Hraw_sorted.astype(image.dtype)
        im_out[:, :, channel][~foreground_indices] = image[:, :, channel][~foreground_indices]

    return im_out.squeeze(), OA


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
            new_pixel = MatlabOperators.round(tim[yy,xx])
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
        x_min = np.min(arr);
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


def rgb2gray(image: Union[np.ndarray, Image.Image], conversion_type: Union[str] = 'equal') -> np.ndarray:
    """
    Convert an RGB image to grayscale (luma, Y′) using ITU luma coefficients.

    Parameters
    ----------
    image (np.ndarray or Image.Image):
        RGB image array with last dimension = 3. Assumed to be gamma-encoded R′G′B′ (i.e., *not* linear light), which
        matches typical sRGB/Rec.709-style images loaded from files.
    conversion_type : {"equal", "rec601", "rec709", "rec2020"}, default "rec709"
        Choice of luma standard:
          - "equal" → Y′ = 0.333 R′ + 0.333 G′ + 0.333 B′
          - "rec601" → Y′ = 0.299 R′ + 0.587 G′ + 0.114 B′
          - "rec709" → Y′ = 0.2125 R′ + 0.7154 G′ + 0.0721 B′
          - "rec2020" → Y′ = 0.2627 R′ + 0.6780 G′ + 0.0593 B′

    Returns
    -------
    gray : np.ndarray
        Grayscale image (same shape as input but last channel removed).

    Notes
    -----
    - This computes luma (Y′) from *gamma-encoded* components, as defined by the ITU
      matrices for Y′CbCr / Y′CbcCrc. For physically linear luminance, you would need
      to first linearize R′G′B′ using the appropriate transfer function,
      mix with linear-light coefficients, then re-encode if desired.

    Args:
        image:
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError(f"Invalid image type {type(image)}. Supported values are Image.Image and np.ndarray")
    ct = ['equal'] if conversion_type.lower() == 'equal' else re.findall(r'\d+', conversion_type)
    if np.sum([c not in conversion_type for c in ['709', '601', '2020', 'equal']])==1 or len(ct) == 0:
        raise ValueError('Conversion type must be either 709, 601, 2020, equal')

    if image.ndim > 2:
        return np.dot(image[..., :3].astype(np.float64), RGB2GRAY_WEIGHTS[ct[0]])
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


def separate(mask: np.ndarray, background: Union[int, float] = None, smoothing: bool = False, show_figure: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Function for simple figure-ground segregation.
    Args:
        mask (np.ndarray): Source mask. Could be an image or a bit mask.
        background (Optional[Union[uint8, float64]]); uint8 value of the background ([0,255]) (e.g., 255)
            or float64 value of the background ([0,1]) (e.g., 1); if equals to 300 or not specified, it is the
            value that occurs the most frequently in mask.
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

    mask = rgb2gray(mask)
    mask = mask.astype(np.float64)/255 if np.max(mask) > 1 else mask
    background = background/255 if 1 < background < 256 else background

    if background == 300:
        # Use np.unique to get unique values and their counts
        unique_values, counts = np.unique(mask.flatten(), return_counts=True)
        background = unique_values[np.argmax(counts)]

    mask_bgr = mask == background

    # Apply median filter to smooth the mask
    if smoothing:
        mask_bgr = apply_median_blur(np.uint8(mask_bgr))
    mask_fgr = (mask_bgr * -1) + 1

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


def gaussian_kernel(size: int, sigma: float, n_dim: int = 2) -> np.ndarray:
    """Generate a normalized Gaussian kernel.

    Args:
        size (int): Size of the kernel. Must be odd.
        sigma (float): Standard deviation of the Gaussian.
        n_dim (int, optional): Dimensionality of the kernel.
            * 1 -> return 1D Gaussian kernel of shape (size,)
            * 2 -> return 2D Gaussian kernel of shape (size, size)
            Defaults to 2.

    Returns:
        np.ndarray: Normalized Gaussian kernel.

    Raises:
        ValueError: If `size` is not odd or `dim` is not 1 or 2.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if n_dim not in (1, 2):
        raise ValueError("n_dim must be 1 or 2.")

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


def ssim_sens(image1: np.ndarray, image2: np.ndarray, n_bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Structural Similarity Index (SSIM) and its gradient.

    Args:
        image1 (np.ndarray): First image as a 3D array.
        image2 (np.ndarray): Second image as a 3D array.
        n_bins (int, optional): Dynamic range of pixel values. Defaults to 256.

    Returns:
        Tuple[np.ndarray, float]:
            - Gradient of SSIM (sensitivity) as a 2D array.
            - Mean SSIM value as a float.

    References :
        1. Avanaki, A.N. Exact global histogram specification optimized for structural similarity.
        OPT REV 16, 613–621 (2009). https://doi.org/10.1007/s10043-009-0119-z

        2. Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment:
        from error visibility to structural similarity," in IEEE Transactions on Image Processing,
        vol. 13, no. 4, pp. 600-612, April 2004, doi: 10.1109/TIP.2003.819861.
    """
    image_x_3D = im3D(image1.astype(np.float64))
    image_y_3D = im3D(image2.astype(np.float64))
    eps = 1e-12  # smallest gradient for covariance computation

    # Gaussian kernel parameters
    window_size = 11
    sigma = 1.5
    window = gaussian_kernel(window_size, sigma, n_dim=1)  # 1d kernel to enable faster convolution with 2d separable kernel trick.

    # Constants for SSIM
    L = n_bins - 1
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Mean calculations
    all_sens, all_mssim = [], []
    for channel in range(image_x_3D.shape[2]):
        # Select channels
        image_x = image_x_3D[:, :, channel]
        image_y = image_y_3D[:, :, channel]

        # Mean pixel intensity
        mu_x = convolve_2d(image_x, window)
        mu_y = convolve_2d(image_y, window)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_x_y = mu_x * mu_y

        # Variances et covariance
        sigma_x_sq = convolve_2d(image_x ** 2, window) - mu_x_sq
        sigma_y_sq = convolve_2d(image_y ** 2, window) - mu_y_sq
        sigma_x_y = convolve_2d(image_x * image_y, window) - mu_x_y

        # Clamp variance and covariance to avoid floating-point (negative) artifacts, which will produce NaNs, Infs,
        # or SSIM > 1
        sigma_x_sq = np.maximum(sigma_x_sq, 0.0)
        sigma_y_sq = np.maximum(sigma_y_sq, 0.0)
        sigma_x_y = np.clip(sigma_x_y, -np.sqrt(sigma_x_sq * sigma_y_sq) - eps, np.sqrt(sigma_x_sq * sigma_y_sq) + eps)

        # SSIM map (Eq. 6)
        num_1 = 2 * mu_x_y + C1
        num_2 = 2 * sigma_x_y + C2
        num = (num_1 * num_2)

        den_1 = mu_x_sq + mu_y_sq + C1
        den_2 = sigma_x_sq + sigma_y_sq + C2
        den = (den_1 * den_2)

        ssim_map = num / den
        mssim = np.mean(ssim_map)

        # SSIM gradient - Eqs. (7) and (8) (Avanaki, 2009)
        term_1 = num_1 / den
        sens = convolve_2d(term_1, window) * image_x

        term_2 = -ssim_map/den_2
        sens += convolve_2d(term_2, window) * image_y

        term_3 = (mu_x * (num_2 - num_1) - mu_y * ssim_map * (den_2 - den_1)) / den
        sens += convolve_2d(term_3, window)

        sens *= 2 / sens.size

        # FIX: scales the SSIM gradient to compensate for its attenuation at larger n_bins and keep the
        # effective update weight consistent across bit depths (more bins require proportionally larger
        # changes for the same effect).
        # sens *= 256**(2*(np.log(n_bins) / np.log(256)-1))

        all_sens.append(sens)
        all_mssim.append(mssim)
    return np.stack(all_sens, axis=-1).squeeze(), np.stack(all_mssim)


def hist_match_validation(images: ImageListType) -> Tuple[np.ndarray, np.ndarray]:

    def normalize_hist(a_hist):
        a_hist = np.float64(a_hist)
        return a_hist / (a_hist.sum(axis=0, keepdims=True) + 1e-12)

    initial_hist = []
    target_hist = np.zeros((images.drange[-1]+1, images[0].ndim))
    for idx, image in enumerate(images):
        initial_hist.append(imhist(image))
        target_hist += initial_hist[-1]
        initial_hist[-1] = normalize_hist(initial_hist[-1])
    target_hist /= len(images)
    target_hist = normalize_hist(target_hist)

    # Compute metric
    N = len(initial_hist)
    corr, rms = np.zeros((N,)), np.zeros((N,))
    for idx, a_hist in enumerate(initial_hist):
        corr[idx] = np.corrcoef(a_hist.ravel(), target_hist.ravel())[0, 1]
        rms[idx] = compute_rmse(a_hist.ravel(), target_hist.ravel())
    return corr, rms


def sf_match_validation(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean magnitude per radius bin (annular average)."""

    def rot_avg(arr2d: np.ndarray, radius: np.ndarray, ann_counts: np.ndarray) -> np.ndarray:
        """Mean magnitude per radius bin (annular average)."""
        sums = np.bincount(radius, weights=arr2d.ravel())
        return sums / ann_counts

    x_size, y_size = images[0].shape[:2]
    n_channels = 3

    # Compute spectra and mean spectrum
    magnitudes, phases = get_images_spectra(images=images)
    target_spectrum = np.zeros(images[0].shape)
    for idx, mag in enumerate(magnitudes):
        target_spectrum += mag
    target_spectrum /= len(magnitudes)
    target_spectrum = im3D(target_spectrum)

    #  Returns the frequencies of the image, bins range from -0.5f to 0.5f (0.5f is the Nyquist frequency) 1/y_size is the distance between each pixel in the image
    f_cols = np.fft.fftshift(np.fft.fftfreq(y_size, d=1 / y_size))  # like f1 in MATLAB
    f_rows = np.fft.fftshift(np.fft.fftfreq(x_size, d=1 / x_size))  # like f2 in MATLAB
    XX, YY = np.meshgrid(f_cols, f_rows)
    nyquistLimit = np.floor(max(x_size, y_size) / 2)
    r, theta = cart2pol(XX, YY)

    # Map of the bins of the frequencies
    r = np.round(r, decimals=0)

    # Need to be a 1D array of integers for the bincount function
    r_int = r.astype(np.int32)
    r1 = r_int.ravel()

    # Precompute counts per radius (for true rotational *averages*, not sums)
    ann_counts = np.bincount(r1)
    ann_counts[ann_counts == 0] = 1  # protect against divide-by-zero
    target_rot_avg = []
    initial_rot_avg = []
    for idx, image in enumerate(images):
        magnitude = im3D(magnitudes[idx])
        phase = im3D(phases[idx])
        tra = []
        ira = []
        for channel in range(n_channels):
            fft_image = magnitude[:, :, channel]

            # Rotational averages (target vs source) as MEANS over annuli
            tra.append(rot_avg(target_spectrum[:, :, channel], radius=r1, ann_counts=ann_counts))
            ira.append(rot_avg(fft_image, radius=r1, ann_counts=ann_counts))
        target_rot_avg.append(np.stack(tra))
        initial_rot_avg.append(np.stack(ira))

    # Compute metrics
    N = len(initial_rot_avg)
    corr, rms = np.zeros((N,)), np.zeros((N,))
    for idx, ira in enumerate(initial_rot_avg):
        corr[idx] = np.corrcoef(ira.ravel(), target_rot_avg[idx].ravel())[0, 1]
        rms[idx] = compute_rmse(ira.ravel(), target_rot_avg[idx].ravel())
    return corr, rms


def spec_match_validation(images: ImageListType) -> Tuple[np.ndarray, np.ndarray]:
    magnitudes, phases = get_images_spectra(images=images)

    target_spectrum = np.zeros(images[0].shape)
    for idx, mag in enumerate(magnitudes):
        target_spectrum += mag
    target_spectrum /= len(magnitudes)

    # Compute metric
    N = len(magnitudes)
    corr, rms = np.zeros((N,)), np.zeros((N,))
    for idx, a_mag in enumerate(magnitudes):
        corr[idx] = np.corrcoef(a_mag.ravel(), target_spectrum.ravel())[0, 1]
        rms[idx] = compute_rmse(a_mag.ravel(), target_spectrum.ravel())
    return corr, rms


def compute_rmse(image1: np.ndarray, image2: np.ndarray) -> float:
    """ Compute the root-mean-square error between two images. """
    return np.sqrt(np.mean((image1 - image2) ** 2))


def get_images_spectra(images: ImageListType, magnitudes: Optional[ImageListType] = None, phases: Optional[ImageListType] = None, rescale: bool = True) -> Union[List[np.ndarray], ImageListType]:
    """
    Get spectrum over list of images
    Args:
        images (ImageListType): List of images.
        magnitudes (Optional[ImageListType]): If provided, inserts new magnitudes into this list.
        phases (Optional[ImageListType]): If provided, inserts new phases into this list.
        rescale (bool): Determines if input is stretched to [0, 1] range.

    Returns:
        magnitudes, phases (Union[List[np.ndarray], ImageListType])

    """
    n_images = len(images)
    x_size, y_size = images.reference_size[:2]
    n_channels = 3 if images.n_dims == 3 else 1
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


def rescale_images(images: ImageListType, rescaling_option: Literal[0, 1, 2, 3] = 2, legacy_mode: bool = False) -> ImageListType:
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

        Returns :
            A list of rescaled images
    """

    # TODO: Shouldn't the input be a uint8 or at least in the [0 to 255] range already?
    if rescaling_option not in [0, 1, 2, 3]:
        raise ValueError(f'The rescaling option must be either [0, 1, 2, 3], now rescaling is : {rescaling_option}')

    n_images = len(images)
    minimum_values = np.zeros((n_images,))
    maximum_values = np.zeros((n_images,))
    for idx, image in enumerate(images):
        minimum_values[idx], maximum_values[idx] = np.min(image), np.max(image)

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
        image (np.ndarray):

    Returns:
        image (np.ndarray) with 3D

    """
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
    mask = np.ones(image.shape).astype(bool) if mask is None else mask.astype(bool)
    mask = np.stack((mask, ) * image.shape[-1], axis=-1) if mask.ndim < image.ndim else im3D(mask)
    n_channels = image.shape[-1]
    count = np.zeros((n_bins, n_channels))
    for channel in range(n_channels):
        count[:, channel], _ = np.histogram(image[:, :, channel][mask[:, :, channel]], bins=n_bins, range=(0, n_bins))

        if normalized:
            count[:, channel] = count[:, channel] / count[:, channel].sum()

    return count