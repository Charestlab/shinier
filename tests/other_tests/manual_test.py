from shinier import ImageDataset, ImageProcessor, Options, utils, ImageListIO
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from shinier.utils import imhist_plot, imhist

from pathlib import Path
import os
import numpy as np
from PIL import Image
from shinier.color import Converter


# conv = Converter.ColorConverter()
# im1 = []
# for i in range(0, 255, 5):
#     for j in range(0, 255, 5):
#         for k in range(0, 3):
#             im1.append([i, j, k])
# im1 = np.array(im1)
# im1 = im1[:-3, :].reshape((78, 100, 3))
# im2 = (np.random.rand(78, 100, 3) * 255).astype(np.uint8)
# images = [im1, im2]

GAMUT_STRATEGY_TYPE = [
    'constrain_dataset_luminance',
    'constrain_dataset_chrominance',
    'constrain_image_luminance',
    'constrain_image_chrominance',
    'clip'
]
TREATMENT = [
    'lum_match',
    'hist_match',
    'sf_match',
    'spec_match'
]
for idx, treatment in enumerate(TREATMENT):
    mode = idx+1
    if mode in [3, 4]:  # [1, 2, 3, 4]:
        for gamut_strategy in GAMUT_STRATEGY_TYPE:
            opts = Options(
                input_folder='/Users/ndr/GIT_REPO/GITHUB/shinier/src/shinier/data/INPUT/',
                output_folder=f"/Users/ndr/GIT_REPO/GITHUB/shinier/src/shinier/data/OUTPUT/{treatment}/{gamut_strategy}/",
                conserve_memory=True,
                as_gray=False,
                linear_luminance=False,
                mode=mode,
                target_hist='equal',
                gamut_strategy=f'{gamut_strategy}',
            )
            print(f"\n\nMode = {mode}, GAMUT_STRATEGY = {gamut_strategy}")
            ds = ImageDataset(options=opts)
            proc = ImageProcessor(dataset=ds, verbose=3)

proc.dataset.images[0].min()

# Convert from sRGB → xyY (internally handles gamma decoding)
_image = conv.sRGB_to_xyY(im1 / 255)

# Extract luminance (Y) channel — for processing
Y = _image[:, :, 2] * 255
# imhist_plot(Y)
Y2 = np.ones(Y.shape)

# Optionally store x and y for later reconstruction
xy = _image[:, :, :2]

xyY = np.dstack([xy, Y2])
rgb2 = conv.xyY_to_sRGB(xyY) * 255


# Define the input folder
input_folder = "/Users/ndr/GIT_REPO/GITHUB/shinier/src/shinier/data/INPUT/"
output_folder = "/Users/ndr/GIT_REPO/GITHUB/shinier/src/shinier/data/OUTPUT/"
im1 = np.random.randint(0, 255, (1024, 1024, 3))
im2 = np.random.randint(0, 255, (1024, 1024, 3))
input_data = [im1, im2]
images = ImageListIO(input_data=input_data, conserve_memory=True)

out = imhist_plot(images[0])
opts = Options(
    output_folder=output_folder,
    conserve_memory=True,
    as_gray=False,
    linear_luminance=False,
    mode=2,
)

ds = ImageDataset(images=images, options=opts)

proc = ImageProcessor(dataset=ds, verbose=4)
out = imhist_plot(ds.images[0])
out = imhist_plot(proc._final_buffer[0])
out = imhist_plot(proc.dataset.images[0])

# Collect all .png files from the folder
png_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]

# Load images as numpy arrays
images = []
images01 = []
for file in png_files:
    file_path = os.path.join(input_folder, file)
    img = Image.open(file_path).convert("RGB")  # convert to RGB to avoid mode issues
    images.append(np.array(img))
    images01.append(np.array(img)/255)

# a = utils.ImageListIO(images, conserve_memory=True)
b = ImageListIO(input_data=images, conserve_memory=False)
initial_hist = []
target_hist = None
for idx, image in enumerate(b):
    initial_hist.append(utils.imhist(image))
    if idx == 0:
        target_hist = initial_hist[-1]
    else:
        target_hist += initial_hist[-1]

target_hist /= (target_hist.sum(axis=0, keepdims=True) + 1e-12)

# Compute spectra and mean spectrum
magnitudes, phases = utils.get_images_spectra(images=b)
target_spectrum = np.zeros(b[0].shape)
for idx, mag in enumerate(magnitudes):
    target_spectrum += mag
target_spectrum /= len(magnitudes)

# Luminance matching
# inp = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/assets/tmp/shard0-of-1/master/case-4d59c1014b72/INPUT"
inp = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/INPUT/"
# oup = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/assets/tmp/shard0-of-1/master/case-4d59c1014b72"
oup = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/MASK/"
masks_folder = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/MASK_64X64')
# combo = {'input_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/assets/SAMPLE_64X64'), 'output_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/OUTPUT'), 'masks_folder': masks_folder, 'whole_image': 1, 'background': 300, 'mode': 3, 'as_gray': True, 'linear_luminance': 0, 'rec_standard': 1, 'dithering': 0, 'conserve_memory': False, 'seed': None, 'legacy_mode': True, 'safe_lum_match': False, 'target_lum': (0, 0.0), 'hist_optim': True, 'hist_specification': 1, 'hist_iterations': 3, 'target_hist': None, 'rescaling': 0, 'target_spectrum': 'unit_test', 'iterations': 1, 'verbose': -1}
# combo = {'input_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/assets/SAMPLE_64X64'), 'output_folder': Path('/Users/ndr/GIT_REPO/GITHUB/shine/OUTPUT'), 'masks_folder': masks_folder, 'whole_image': 1, 'background': 300, 'mode': 2, 'as_gray': True, 'linear_luminance': 0, 'rec_standard': 1, 'dithering': 0, 'conserve_memory': False, 'seed': None, 'legacy_mode': True, 'safe_lum_match': False, 'target_lum': (0, 0.0), 'hist_optim': True, 'hist_specification': 1, 'hist_iterations': 3,'target_hist': None, 'rescaling': 0, 'target_spectrum': None, 'iterations': 1, 'verbose': 3}
# bm = ImageListIO(input_data=masks_folder)
# my_options = Options(**combo)
my_options = Options(
    input_folder=inp,
    output_folder=oup,
    masks_folder=None,
    whole_image=1,
    background=300,
    mode=5,
    as_gray=True,
    linear_luminance=0,
    rec_standard=2,
    dithering=1,
    conserve_memory=True,
    seed=None,
    legacy_mode=False,
    safe_lum_match=False,
    target_lum=(0, 0),
    hist_optim=False,
    hist_specification=4,
    hist_iterations=5,
    target_hist=None,
    rescaling=0,
    target_spectrum=None,
    iterations=5,
    verbose=3,
)
my_dataset = ImageDataset(options=my_options)
results = ImageProcessor(dataset=my_dataset, verbose=3)
fig = utils.show_processing_overview(results, img_idx=0)


images = results.get_results()
_, rmse_hist_before = utils.hist_match_validation(images=results._initial_buffer, binary_masks=results.bool_masks, target_hist=results._target_hist)
_, rmse_hist_after = utils.hist_match_validation(images=results._final_buffer, binary_masks=results.bool_masks, target_hist=results._target_hist)

_, rmse_sf_before = utils.sf_match_validation(images=results._initial_buffer, target_spectrum=results._target_spectrum)
_, rmse_sf_after = utils.sf_match_validation(images=results._final_buffer, target_spectrum=results._target_spectrum)

corr1a, rmse1a = utils.hist_match_validation(images=b)
corr1b, rmse1b = utils.sf_match_validation(images=b)
corr1c, rmse1c = utils.spec_match_validation(images=b)
my_dataset = ImageDataset(options=my_options)
results = ImageProcessor(dataset=my_dataset, verbose=2)
corr2a, rmse2a = utils.hist_match_validation(images=results.dataset.images)
corr2b, rmse2b = utils.sf_match_validation(images=results.dataset.images)
corr2c, rmse2c = utils.spec_match_validation(images=results.dataset.images)

# target_hist = np.ones((256,3))/(256*3)
# plt.imshow(my_dataset.images[0], cmap='gray')
# plt.imshow(results.dataset.images[-3], cmap='gray')

new_images = results.get_results()
plt.imshow(new_images[0])
my_options = Options(
    input_folder='./../INPUT/',
    output_folder='./../OUTPUT',
    masks_folder='./../MASK/',
    whole_image=1,
    legacy_mode=1,
    conserve_memory=False,
    as_gray=True,
    mode=1,
    safe_lum_match = True,
    dithering = True,
)

plt.imshow(results.images[-1], cmap='gray')

