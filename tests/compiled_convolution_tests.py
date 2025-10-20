import numpy as np
import time
import shinier.utils as u  # import the actual module

# ---- benchmark ----
rng = np.random.default_rng(0)
img = rng.random((1024, 1024))
g = np.array([1, 4, 6, 4, 1], float); g /= g.sum()
K = np.outer(g, g)  # 2D Gaussian-like kernel

for ker in (g, K):
    # --- Python fallback ---
    u._HAS_CYTHON = False
    t0 = time.perf_counter()
    ref = u.convolve_2d(img, ker)
    t1 = time.perf_counter()

    # --- Cython backend ---
    u._HAS_CYTHON = True
    t2 = time.perf_counter()
    out = u.convolve_2d(img, ker)
    t3 = time.perf_counter()

    diff = np.abs(out - ref)
    print(f"{ker.ndim}D kernel:")
    print(f"  max={diff.max():.3e}, mean={diff.mean():.3e}")
    print(f"  Fallback: {t1 - t0:.3f}s, Cython: {t3 - t2:.3f}s, speedup={(t1 - t0)/(t3 - t2):.1f}×\n")



# pip install -e .
# python -X faulthandler "tests/compiled_convolution_tests.py"