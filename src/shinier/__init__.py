"""
SHINIER: Spectrum, Histogram, and Intensity Normalization, Equalization, and Refinement.

This package provides advanced image-processing utilities for luminance,
histogram, and spatial frequency normalization, adapted from the original
MATLAB SHINE Toolbox.

References:
    Willenbockel, V., Sadr, J., Fiset, D., Horne, G. O., Gosselin, F., & Tanaka, J. W. (2010).
    Controlling low-level image properties: The SHINE toolbox.
    *Behavior Research Methods, 42*(3), 671–684. https://doi.org/10.3758/BRM.42.3.671

    See accompanying paper: Salvas-Hébert, M.*, Dupuis-Roy, N.*, Landry, C., Charest, I., & Gosselin, F. (2026).
    SHINIER: An open-source Python package for controlling low-level image properties.
    SoftwareX, 35, Article 102884. https://doi.org/10.1016/j.softx.2026.102884
"""

# Metadata
__author__ = "Nicolas Dupuis-Roy and Mathias Salvas-Hebert"
__version__ = "0.2.2"
__email__ = "nicolas.dupuis.roy@umontreal.ca"

# For direct importation
from importlib import util
from pathlib import Path
import sys
import warnings

_HAS_CYTHON = False
convolve2d_direct = None
convolve2d_separable = None

# This is the *package* root: src/shinier in dev, site-packages/shinier when installed
DEV_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parent

if util.find_spec("shinier._cconvolve") is not None:
    try:
        from ._cconvolve import convolve2d_direct, convolve2d_separable
        _HAS_CYTHON = True
    except Exception as exc:
        try:
            import numpy as _np

            numpy_version = _np.__version__
        except Exception:
            numpy_version = "unavailable"

        warnings.warn(
            "SHINIER could not load the optional compiled convolution extension "
            "(`shinier._cconvolve`). SHINIER will keep working, but convolution-heavy "
            "operations will use the slower NumPy fallback. "
            f"Python: {sys.version.split()[0]}; NumPy: {numpy_version}. "
            "If you want the faster compiled extension, try reinstalling SHINIER after "
            "upgrading pip, setuptools, wheel, and NumPy, and make sure a C++ compiler "
            f"is available. Original error: {exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )

__all__ = [
    "Options",
    "ImageDataset",
    "ImageListIO",
    "ImageProcessor",
    "convolve2d_direct",
    "convolve2d_separable",
    "StimulusMasker",
    "_HAS_CYTHON",
    "color",
    "SHINIER_CLI",
    "REPO_ROOT",
    "DEV_ROOT",
    "__version__",
]

from .Options import Options
from .ImageDataset import ImageDataset
from .ImageListIO import ImageListIO
from .ImageProcessor import ImageProcessor
from .SHINIER import SHINIER_CLI
from .utils import StimulusMasker, imstats, ImageStats
from . import color
