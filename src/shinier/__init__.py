"""
Shinier: Scientific Histogram Normalization and Image Equalization in R/G/B.

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
__version__ = "0.1.0"
__email__ = "n.dupuis.roy@gmail.com"

# For direct importation
from importlib import util
_HAS_CYTHON = util.find_spec("shinier._cconvolve") is not None

if _HAS_CYTHON:
    from ._cconvolve import convolve2d_direct, convolve2d_separable
else:
    convolve2d_direct = None
    convolve2d_separable = None

from .Options import Options
from .ImageDataset import ImageDataset
from .ImageListIO import ImageListIO
from .ImageProcessor import ImageProcessor


__all__ = [
    "Options",
    "ImageDataset",
    "ImageListIO",
    "ImageProcessor",
    "convolve2d_direct",
    "convolve2d_separable",
    "_HAS_CYTHON",
]