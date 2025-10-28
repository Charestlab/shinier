"""
Setup script for the SHINIER package.

This script builds an optional Cython/C++ extension (_cconvolve) that accelerates
2D and separable convolution routines. If the C++ file exists, it is used directly
(no Cython required). If not, a .pyx file is compiled locally.

If compilation fails (e.g., missing compiler or OpenMP), the pure NumPy fallback
in `utils.py` will be used automatically at runtime.
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from pathlib import Path
import sys
import numpy


class build_ext(_build_ext):
    """Ensure NumPy headers are available before building extensions."""
    def build_extensions(self):
        for ext in self.extensions:
            ext.include_dirs.append(numpy.get_include())
        super().build_extensions()


def make_cconvolve_ext() -> Extension:
    """Configure and return the _cconvolve extension (Cython/C++)."""
    base = Path("src/shinier/_cconvolve")
    pyx_file = base.with_suffix(".pyx")
    cpp_file = base.with_suffix(".cpp")

    # Prefer pre-generated C++ for user installations
    if cpp_file.exists():
        sources = [str(cpp_file)]
    elif pyx_file.exists():
        sources = [str(pyx_file)]
    else:
        raise FileNotFoundError("Missing both _cconvolve.cpp and _cconvolve.pyx")

    # Base optimization flags
    compile_args = ["-O3"]
    link_args = []

    # Platform-specific handling
    if sys.platform.startswith("linux"):
        compile_args += ["-fopenmp"]
        link_args += ["-fopenmp"]
    elif sys.platform == "win32":
        compile_args = ["/O2", "/openmp"]
    elif sys.platform == "darwin":
        # macOS: skip OpenMP by default for compatibility (Clang lacks it by default)
        pass  # Keep only -O3
    else:
        compile_args += ["-O3"]

    return Extension(
        name="shinier._cconvolve",
        sources=sources,
        language="c++",  # consistent with generated .cpp
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )


setup(
    name="shinier",
    version="0.1.0",
    description="A Python package for the SHINIER toolbox (image property normalization)",
    author="Nicolas Dupuis-Roy",
    author_email="n.dupuis.roy@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.1",
        "pillow>=9.0.1",
        "matplotlib>=3.9.2",
    ],
    python_requires=">=3.9, <4",
    ext_modules=[make_cconvolve_ext()],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)