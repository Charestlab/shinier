from shinier import ImageDataset, ImageProcessor, Options, utils, ImageListIO
from matplotlib import pyplot as plt
from shinier.utils import imhist_plot

import os
import numpy as np
from PIL import Image

# Define the input folder
input_folder = "/Users/ndr/GIT_REPO/GITHUB/shine/shinier/INPUT/"

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
b = ImageListIO(images, conserve_memory=False)
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
my_options = Options(
    input_folder="/Users/ndr/GIT_REPO/GITHUB/shine/shinier/INPUT/",
    images_format='png',
    whole_image=1,
    iterations=3,
    legacy_mode=False,
    conserve_memory=True,
    as_gray=0,
    mode=2,
    dithering=0,
    hist_iterations=3,
    hist_optim=1,
    hist_specification=4,
    target_hist=None
)
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
    images_format="png",
    masks_format="png",
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

