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
```base
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

Example use case: the "luminance" will be ajusted so that the mean values and the standard
 deviation of the output images will be the average if the input images. The 
 "safe_lum_match" setting being off will cause some values to be clipped and 
 set to either 0 (< 0) or 255 (> 255).
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 1,
    #safe_lum_match = False,
    #target_lum = (0, 0)  # (mean, std), here it's the average of the images
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

Example use case: Exact histogram specification (no noise), with SSIM
  optimization enabled. After spectrum alignment. Rescaling is done by 
  default.
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 6,
    hist_optim = 1           # enable SSIM optimization
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
from PIL import Image
from shinier.utils import cart2pol, image_spectrum
import numpy as np

im = Image.open("mon_image.png").convert("L")
im = np.array(im, dtype=np.float64)
target_spectrum = image_spectrum(im)[0]

opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    mode = 8,
    hist_specification = 4,  # Hybrid algorithm
    target_spectrum = target_spectrum
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
    mode = 9
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
  value in the most will be automatically selected using the most frequent
  grayscale value in the masks, the image will be transform to a grayscale image
  (1 channel), the dithering won't be applied before saving, the smart memory
  gestion won't be used here and legacy_mode, which is to replicate MATLAB more
  closely is used.
"""
opts = Options(
    input_folder=INPUT_FOLDER,
    output_folder=OUTPUT_FOLDER,
    masks_folder=MASKS_FOLDER,
    mode = 2,                # hist_match
    whole_image = 3,         # figure-ground separation using masks
    background = 300,        # masking value: most frequent grayscale value
    as_gray = True,          # RGB or grayscale
    dithering = False,       # noisy-bit dithering
    conserve_memory = False, # smart memory management
    legacy_mode = True       # use MATLAB-like operators (e.g., round)
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
dataset = ImageDataset(images=ImageListIO(im_loaded_before), options=opts)
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

fig = show_processing_overview(results)
plt.show()
```

---

## Thank you

Thank you for taking the time to explore these demos.

If you’d like more information about the available options or a deeper understanding of how each method works, you can refer to the **docstrings in the classes and functions**. All core components of **SHINIER** are thoroughly documented, and the in-code descriptions explain the algorithms, parameters, and expected behavior in detail.

Feel free to dig into the source — the docstrings are designed to guide you step by step.
