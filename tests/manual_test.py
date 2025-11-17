from shinier import ImageDataset, ImageProcessor, Options, utils, ImageListIO
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from shinier.utils import imhist_plot

from pathlib import Path
import os
import numpy as np
from PIL import Image


# inp = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/src/shinier/data/INPUT/"
# oup = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/src/shinier/data/OUTPUT/"
# mask_folder = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/src/shinier/data/MASK/"

oup = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/src/shinier/data/OUTPUT/"
masks_folder = '/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/MASK_64X64'
combo = {'input_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/SAMPLE_64X64'), 'output_folder': Path(oup), 'masks_folder': Path(masks_folder), 'whole_image': 1, 'background': 300, 'mode': 2, 'seed': None, 'legacy_mode': True, 'iterations': 5, 'as_gray': True, 'linear_luminance': True, 'rec_standard': 1, 'dithering': 0, 'conserve_memory': True, 'safe_lum_match': True, 'target_lum': (0, 0), 'hist_optim': True, 'hist_specification': 1, 'hist_iterations': 3, 'target_hist': 'unit_test', 'rescaling': 3, 'target_spectrum': None, 'verbose': -1}
# combo = {'input_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/SAMPLE_64X64'), 'output_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/OUTPUT'), 'masks_folder': masks_folder, 'whole_image': 1, 'background': 300, 'mode': 3, 'as_gray': True, 'linear_luminance': 0, 'rec_standard': 1, 'dithering': 0, 'conserve_memory': False, 'seed': None, 'legacy_mode': True, 'safe_lum_match': False, 'target_lum': (0, 0.0), 'hist_optim': True, 'hist_specification': 1, 'hist_iterations': 3, 'target_hist': None, 'rescaling': 0, 'target_spectrum': 'unit_test', 'iterations': 1, 'verbose': -1}
# combo = {'input_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/SAMPLE_64X64'), 'output_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/OUTPUT'), 'masks_folder': masks_folder, 'whole_image': 1, 'background': 300, 'mode': 2, 'as_gray': True, 'linear_luminance': 0, 'rec_standard': 1, 'dithering': 0, 'conserve_memory': False, 'seed': None, 'legacy_mode': True, 'safe_lum_match': False, 'target_lum': (0, 0.0), 'hist_optim': True, 'hist_specification': 1, 'hist_iterations': 3,'target_hist': None, 'rescaling': 0, 'target_spectrum': None, 'iterations': 1, 'verbose': 3}
# bm = ImageListIO(input_data=masks_folder)
my_options = Options(**combo)
# target_hist = np.load('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/target_hist.npy')
# my_options = Options(
#     input_folder=inp,
#     output_folder=oup,
#     masks_folder=None,
#     whole_image=1,
#     background=300,
#     mode=5,
#     as_gray=False,
#     linear_luminance=False,
#     rec_standard=2,
#     dithering=0,
#     conserve_memory=True,
#     seed=None,
#     legacy_mode=False,
#     safe_lum_match=False,
#     target_lum=(0, 0),
#     hist_optim=False,
#     hist_specification=4,
#     hist_iterations=5,
#     target_hist=None,
#     rescaling=0,
#     target_spectrum=None,
#     iterations=1000,
#     verbose=3,
# )
my_dataset = ImageDataset(options=my_options)
results = ImageProcessor(dataset=my_dataset, verbose=3)
fig = utils.show_processing_overview(results, img_idx=0, show_initial_target=True)

