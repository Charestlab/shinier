"""
Python package for shinier

See accompanying paper: XXX
See original paper: Willenbockel V, Sadr J, Fiset D, Horne GO, Gosselin F, Tanaka JW. Controlling low-level image properties: the SHINE toolbox. Behav Res Methods. 2010 Aug;42(3):671-84. doi: 10.3758/BRM.42.3.671. PMID: 20805589.

"""

# Metadata
__author__ = "Nicolas Dupuis-Roy"
__version__ = "0.1.0"
__email__ = "n.dupuis.roy@gmail.com"

# For direct importation
from .Options import Options
from .ImageDataset import ImageDataset
from .ImageListIO import ImageListIO
from .ImageProcessor import ImageProcessor
