"""
SHINIER: Spectrum, Histogram, and Intensity Normalization, Equalization, and Refinement.

This package provides advanced image-processing utilities for luminance,
histogram, and spatial frequency normalization, adapted from the original
MATLAB SHINE Toolbox.

References:
    Willenbockel, V., Sadr, J., Fiset, D., Horne, G. O., Gosselin, F., & Tanaka, J. W. (2010).
    Controlling low-level image properties: The SHINE toolbox.
    *Behavior Research Methods, 42*(3), 671–684. https://doi.org/10.3758/BRM.42.3.671

    See accompanying paper: Salvas-Hébert, M., Dupuis-Roy, N., Landry, C., Charest, I. & Gosselin, F. (2025)
    The SHINIER the Better: An Adaptation of the SHINE Toolbox on Python.
"""

# Metadata
__author__ = "Nicolas Dupuis-Roy"
__version__ = "0.1.8"
__email__ = "nicolas.dupuis.roy@umontreal.ca"

# For direct importation
from importlib import util
from pathlib import Path
_HAS_CYTHON = util.find_spec("shinier._cconvolve") is not None

# This is the *package* root: src/shinier in dev, site-packages/shinier when installed
DEV_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parent

if _HAS_CYTHON:
    from ._cconvolve import convolve2d_direct, convolve2d_separable
else:
    convolve2d_direct = None
    convolve2d_separable = None

__all__ = [
    "Options",
    "ImageDataset",
    "ImageListIO",
    "ImageProcessor",
    "convolve2d_direct",
    "convolve2d_separable",
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
from . import color
