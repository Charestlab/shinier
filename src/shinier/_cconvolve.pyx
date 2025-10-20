# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython-accelerated 2D convolution with reflect padding.

Public API:
    - convolve2d_direct(img, ker)
    - convolve2d_separable(img, k1d)

Notes:
    * Inputs must be 2D float64 NumPy arrays.
    * "Reflect" padding uses NumPy semantics (no edge duplication).
"""

import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t


# ----------------------------------------------------------------------
# Utility: reflection index (NumPy's reflect semantics, no duplication)
# ----------------------------------------------------------------------
cdef inline Py_ssize_t _reflect_index(Py_ssize_t n, Py_ssize_t i) nogil:
    if n <= 1:
        return 0
    cdef Py_ssize_t period = 2 * (n - 1)
    i = i % period
    if i < 0:
        i += period
    if i >= n:
        i = period - i
    return i


# ----------------------------------------------------------------------
# 1D convolution along one row (horizontal)
# ----------------------------------------------------------------------
cdef void _row_convolve_1d(
    DTYPE_t[:, ::1] img,
    DTYPE_t[::1] k1d,
    DTYPE_t[:, ::1] out,
    Py_ssize_t row
) noexcept nogil:
    cdef Py_ssize_t W = img.shape[1]
    cdef Py_ssize_t K = k1d.shape[0]
    cdef Py_ssize_t r = K // 2
    cdef Py_ssize_t j, t, jj
    cdef DTYPE_t acc

    for j in range(W):
        acc = 0.0
        for t in range(-r, r + 1):
            jj = _reflect_index(W, j + t)
            acc += img[row, jj] * k1d[t + r]
        out[row, j] = acc


# ----------------------------------------------------------------------
# 2D dense convolution (square kernel)
# ----------------------------------------------------------------------
cdef void _dense_convolve_2d(
    DTYPE_t[:, ::1] img,
    DTYPE_t[:, ::1] ker,
    DTYPE_t[:, ::1] out
) noexcept nogil:
    cdef Py_ssize_t H = img.shape[0]
    cdef Py_ssize_t W = img.shape[1]
    cdef Py_ssize_t K = ker.shape[0]
    cdef Py_ssize_t r = K // 2
    cdef Py_ssize_t i, j, u, v, ii, jj
    cdef DTYPE_t acc

    for i in range(H):
        for j in range(W):
            acc = 0.0
            for u in range(-r, r + 1):
                ii = _reflect_index(H, i + u)
                for v in range(-r, r + 1):
                    jj = _reflect_index(W, j + v)
                    acc += img[ii, jj] * ker[u + r, v + r]
            out[i, j] = acc


# ----------------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------------
def convolve2d_direct(
    cnp.ndarray[DTYPE_t, ndim=2] img,
    cnp.ndarray[DTYPE_t, ndim=2] ker
):
    """Convolve a 2D image with a square 2D kernel (reflect padding)."""
    if img.dtype != np.float64 or not img.flags.c_contiguous:
        img = np.ascontiguousarray(img, dtype=np.float64)
    if ker.dtype != np.float64 or not ker.flags.c_contiguous:
        ker = np.ascontiguousarray(ker, dtype=np.float64)

    if ker.shape[0] != ker.shape[1]:
        raise ValueError("2D kernel must be square.")
    cdef Py_ssize_t K = ker.shape[0]
    if K % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] img_arr = img
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] ker_flip = np.ascontiguousarray(ker[::-1, ::-1])
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] out_arr = np.empty_like(img_arr)

    cdef DTYPE_t[:, ::1] imgv = img_arr
    cdef DTYPE_t[:, ::1] krev = ker_flip
    cdef DTYPE_t[:, ::1] outv = out_arr

    with nogil:
        _dense_convolve_2d(imgv, krev, outv)

    return np.asarray(out_arr)


def convolve2d_separable(
    cnp.ndarray[DTYPE_t, ndim=2] img,
    cnp.ndarray[DTYPE_t, ndim=1] k1d
):
    """Apply separable 2D convolution via two 1D passes (reflect padding)."""
    if img.dtype != np.float64 or not img.flags.c_contiguous:
        img = np.ascontiguousarray(img, dtype=np.float64)
    if k1d.dtype != np.float64 or not k1d.flags.c_contiguous:
        k1d = np.ascontiguousarray(k1d, dtype=np.float64)

    cdef Py_ssize_t K = k1d.shape[0]
    if K % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] img_arr = img
    cdef cnp.ndarray[DTYPE_t, ndim=1, mode='c'] k1d_arr = k1d
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] tmp_arr = np.empty_like(img_arr)
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] out_arr = np.empty_like(img_arr)

    cdef DTYPE_t[:, ::1] imgv = img_arr
    cdef DTYPE_t[::1]  k1dv = k1d_arr
    cdef DTYPE_t[:, ::1] tmpv = tmp_arr
    cdef DTYPE_t[:, ::1] outv = out_arr

    cdef Py_ssize_t H = img_arr.shape[0]
    cdef Py_ssize_t W = img_arr.shape[1]
    cdef Py_ssize_t i, j, t, ii, jj
    cdef DTYPE_t acc

    # Pass 1: horizontal
    with nogil:
        for i in range(H):
            _row_convolve_1d(imgv, k1dv, tmpv, i)

    # Pass 2: vertical
    with nogil:
        for j in range(W):
            for i in range(H):
                acc = 0.0
                for t in range(-K // 2, K // 2 + 1):
                    ii = _reflect_index(H, i + t)
                    acc += tmpv[ii, j] * k1dv[t + K // 2]
                outv[i, j] = acc

    return np.asarray(out_arr)