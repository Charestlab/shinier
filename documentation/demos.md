```text
   ███████╗██╗  ██╗██╗███╗  ██╗██╗███████╗██████╗
   ██╔════╝██║  ██║██║████╗ ██║██║██╔════╝██╔══██╗
   ███████╗███████║██║██╔██╗██║██║█████╗  ██████╔╝
   ╚════██║██╔══██║██║██║╚████║██║██╔══╝  ██╔══██╗
   ███████║██║  ██║██║██║ ╚███║██║███████╗██║  ██║
   ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚══╝╚═╝╚══════╝╚═╝  ╚═╝
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![PyPI version](https://img.shields.io/pypi/v/shinier.svg)](https://pypi.org/project/shinier/)
---

# Demos / How-to-use

> The package can be used in two ways:  
> • via the **command-line interface (CLI)**  
> • or by defining the options directly in your **code**  
>
> Below is a summary of all nine available modes. Full descriptions can be found in `Options.py`.

```python
modes:
      1 = lum_match
      2 = hist_match (default)
      3 = safe_lum_match
      4 = spec_match
      5 = hist_match & sf_match
      6 = hist_match & spec_match
      7 = sf_match & hist_match
      8 = spec_match & hist_match
      9 = only dithering
```

---

## Case 1 – Using the CLI

The **CLI** lets you process images interactively.  
If paths are not specified, SHINIER uses:

- the five example images in `input_folder`
- masks (if provided) in `masks_folder`
- automatic saving in `output_folder`

Install SHINIER:

```bash
pip install shinier
```

---
### I. Calling the CLI

#### 1) Recommended: From terminal
##### Calls the CLI
```bash
shinier
```
##### Displays the processing overview of image #1 after CLI
```bash
shinier --show_results --image_index=1
```
##### Save the processing overview of image #0 after CLI
```bash
shinier --show_results --save_path="path/file.png"
```

#### 2) From Python (Not recommended)

```python
from shinier import SHINIER_CLI
SHINIER_CLI()
```

---

### II. CLI Use Cases

#### 1) Press Enter

Use default value:

```text
> Default selected: shinier/INPUT
```

#### 2) Press `q`

Exit:

```text
Exit requested (q).
```

#### 3) Write custom input

Provide strings, numbers, or choices:

```text
Users/.../.../my_input
```

---

### III. CLI Profiles

| Profile   | Description                                                       |
|----------|-------------------------------------------------------------------|
| Default  | Use default parameters                                            |
| Legacy   | Emulates MATLAB SHINE toolbox                                     |
| Custom   | Full manual control over all options                              |

All profiles below use default parameters with the sample images.

---

## Case 2 – Customizing Options

You can bypass the CLI by using an `Options` object.

> *Commented parameters are defaults and do not need to be set.*

```python
from shinier import ImageDataset, ImageProcessor, Options
```

---

### 1) Define the Options

Assuming grayscale images for the examples:

```python
INPUT_FOLDER  = "path"
OUTPUT_FOLDER = "path"
MASKS_FOLDER  = "path"
```

---

### Mode 1 – `lum_match`

```python
"""
Mode 1 (lum_match): simple normalization for the grayscale values of one or
  multiple channel. It adjusts the mean grayscale value and standard-deviation
  for a desired (M, STD).

Example use case: the "luminance" will be adjusted so that the mean values and the standard
 deviation of the output images will be the average of the input images. Turning 
 "safe_lum_match" off can allow some values to be clipped later to 0 (< 0) or 
 255 (> 255).
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 1,
    #safe_lum_match = False,
    #target_lum = (0, 0)  # 0 means dataset average for that statistic
    #target_lum = (0, 20)  # use dataset-average mean, set contrast/std to 20
    #target_lum = (100, 0)  # set mean to 100, use dataset-average contrast/std
    #target_lum = (None, 20)  # keep each image mean, set contrast/std to 20
    #target_lum = (100, None)  # set mean to 100, keep each image contrast/std
)
```

---

### Mode 2 – `hist_match`

```python
"""
Mode 2 (hist_match): matches the luminance histograms of a number of source
  images with a specified target histogram.

Example use case: the histogram matching will be done using Coltuc, Bolon and
  Chassery (2006) technique while optimizing for structural similarity (Avanaki,
  2009) and the target histogram will be the average of the input images.
  verbose at 3 will give you more informations about the processing.
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 2,
    hist_optim = 1,           # Avanaki, 2009
    hist_specification = 2,   # Coltuc, Bolon & Chassery, 2006
    verbose = 3,
    #target_hist = None
)
```

---

### Mode 3 – `sf_match`

```python
"""
Mode 3 (sf_match): matches the rotational average of the Fourier amplitude
  spectra for a set of images.

Example use case: will match the rotational average with the average spectrum of
  all the images since target spectrum is not specified. The grayscale values of
  the images will be then rescaled after the image modification with the option
  #2 (Rescaling absolute max/min — shared 0–1 range).
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 3,
    rescaling = 2,            # rescaling absolute min/max — shared 0–1 range
    #target_spectrum = None
)
```

---

### Mode 4 – `spec_match`

```python
"""
Mode 4 (spec_match): matches the amplitude spectrum of the source image with a
  specified target spectrum.

Example use case: will match the amplitude spectrum of the images with the
  average one of all the images since target spectrum is not specified. The
  grayscale values of the images will then be rescaled after the image
  modification with the option #2 (Rescaling absolute max/min — shared 0–1 range).
"""

opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 4,
    #rescaling = 2,
    #target_spectrum = None
)
```

---

### Mode 5 – `hist_match` → `sf_match`

```python
"""
Mode 5 (hist_match & sf_match): histogram matching followed by rotational
  Fourier spectrum alignment.

Example use case: Histogram specification with noise is applied (legacy method),
  then rotational Fourier spectra are aligned. No rescaling is performed
  afterwards,to preserve the luminance distribution imposed by histogram
  matching.
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 5,
    hist_specification = 1,  # histogram specification with noise (legacy)
    rescaling = 0,            # no rescaling after Fourier alignment
    verbose = 2
)
```

---

### Mode 6 – `hist_match` → `spec_match`

```python
"""
Mode 6 (hist_match & spec_match): histogram matching followed by full Fourier
  spectrum alignment.

Example use case: Exact histogram specification with SSIM optimization is enabled for histogram matching, then
  full Fourier spectra are aligned. Rescaling is done by default.
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 6,
    hist_optim = True,        # enable SSIM optimization
    #rescaling = 2
)
```

---

### Mode 7 – `sf_match` → `hist_match`

```python
"""
Mode 7 (sf_match & hist_match): rotational Fourier spectrum alignment followed
  by histogram matching.

Example use case: Spectrum alignment ensures comparable spatial frequency
  content, then histogram specification is applied with noise. No SSIM
  optimization is performed. Rescaling is skipped.
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 7,
    hist_optim = 0,
    hist_specification = 1,
    rescaling = 0
)
```

---

### Mode 8 – `spec_match` → `hist_match`

```python
"""
Mode 8 (spec_match & hist_match): full Fourier spectrum alignment and histogram
  matching.

Example use case: Spectrum alignment is done with respect to a predefined
  target_spectrum (instead of the average of all input images). Afterwards,
  histogram specification is applied with 'Hybrid' algorithm, and luminance 
  values are rescaled to global min/max.
"""

opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 8,
    hist_specification = 4,  # Hybrid algorithm
    target_spectrum = "img_path.png"
)
```

---

### Mode 9 – only dithering

```python
"""
Mode 9 (only dithering): applies noisy-bit dithering Allard & Faubert, 2008).

Example use case: dithering will be applied with the default noisy-bit method
  (Allard & Faubert, 2008), while leaving the original image luminance and
  spectrum unchanged.
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 9,
    dithering = 1
)
```

---

### Example 10 – Mode 2 + extra parameters

```python
"""
Example 10 (mode 2 + non-mode-specific parameters): to show the other parameters.

Example use case: hist_matching using Coltuc, Bolon & Chassery (2006) exact
  histogram specification. Target histogram will be the average from all the
  images (default), no SSIM optimization (Avanki, 2009).

  The masks are used for figure-ground separation (whole_image = 3), background
  value in the mask will be automatically selected using the most frequent
  grayscale value in the masks, the images will be transformed to grayscale
  (1 channel), the dithering won't be applied before saving, the smart memory
  management won't be used here, and legacy_mode will reproduce MATLAB-like
  defaults where applicable.
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 2,                # hist_match
    whole_image = 3,         # figure-ground separation using masks
    background = 300,        # masking value: most frequent grayscale value
    as_gray = True,          # convert to grayscale
    dithering = 0,           # no dithering
    conserve_memory = False, # smart memory management
    legacy_mode = True       # use MATLAB-like operators (e.g., round)
)
```

---

### Example 11 – Preserve colors with `constrain_image_chrominance`

```python
"""
Example 11: Show how to preserve image colors while modifying histogram.

Use-case: perform spectrum + histogram matching but avoid hue shifts or
out-of-gamut repairs that change perceived colors by applying
`gamut_strategy='constrain_image_chrominance'`.

Notes:
- `linear_luminance=False` enables xyY pipelines (required for gamut strategies).
- `as_gray=False` keeps processing in color (luminance-only transforms applied to Y).
"""
opts = Options(
  input_folder=INPUT_FOLDER,
  output_folder=OUTPUT_FOLDER,
  mode=8,                                      # spec_match -> hist_match
  as_gray=False,                               # keep color
  linear_luminance=False,                      # use xyY conversions
  rec_standard=2,                              # Rec.709 (sRGB-like)
  gamut_strategy='constrain_image_chrominance',# preserve chroma, repair per-image
  dithering=0,                                 # no dithering for diagnostics
  conserve_memory=True,                        # stream-friendly for large datasets
  verbose=2
)
```



---

### 2) Create the Dataset

#### (i) Recommended: from folders

```python
dataset = ImageDataset(options=opts)
```

#### (ii) Manual: from pre-loaded images (not recommended)

```python
from shinier import ImageListIO

im_loaded_before = [...]  # list of numpy arrays
dataset = ImageDataset(images=ImageListIO(input_data=im_loaded_before), options=opts)
```

---

### 3) Image Processing

```python
results = ImageProcessor(dataset=dataset)
# Output images are stored in results.dataset.images
```

#### Optional: display a processing overview

```python
from shinier.utils import show_processing_overview
import matplotlib.pyplot as plt

fig = show_processing_overview(results)
plt.show()
```
--- 

## Thank you

Thank you for taking the time to explore these demos.

If you’d like more information about the available options or a deeper understanding of how each method works, you can refer to the **docstrings in the classes and functions**. All core components of **SHINIER** are thoroughly documented, and the in-code descriptions explain the algorithms, parameters, and expected behavior in detail.

Feel free to dig into the source — the docstrings are designed to guide you step by step.
