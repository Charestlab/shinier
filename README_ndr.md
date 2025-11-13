# 🌟 Shinier: Scientific Histogram Intensity Normalization and Image Equalization in RGB

<p align="center">
  <img src="./assets/SHINIER_logo.jpg" width="400" alt="SHINIER Logo">
</p>

## 🎯 Overview

**SHINIER** is a modern Python implementation of the **SHINE** (Spectrum, Histogram, and Intensity Normalization and Equalization) toolbox, originally developed in MATLAB by [Willenbockel et al. (2010)](https://doi.org/10.3758/BRM.42.3.671).  

It enables precise control of low-level visual properties — luminance, contrast, histogram, and spectral content — across large image sets for well-calibrated visual experiments and related analyses.

This new version introduces major improvements:

- 🎨 **Color Processing** — New modes for color image control with modern color-space standards (Rec.601 / Rec.709 / Rec.2020).
- 🖼️ **Dithering Support** — Reduces quantization artifacts and enhances output image quality.
- ⚡ **Optimized Performance** — Efficient memory management and faster processing for large image sets.
- 🔢 **High-Precision Arithmetic** — Computations in floating-point precision rather than 8-bit integer space, minimizing rounding errors in multi-stage processing.  
- 📦 **Object-Oriented Design** — Modular, extensible architecture with clean API and CLI interfaces.  
- 🕰 **Legacy Mode** — Ensures full backward compatibility with MATLAB’s original SHINE toolbox.


## 🚀 Quick Start

### Installation

##### Install from PyPI (recommended):

```bash
pip install shinier
```
> **Note:** SHINIER includes a Cython-compiled C++ extension (`_cconvolve`) for faster convolution. If a C/C++ compiler is available, it will build automatically during installation, otherwise, it will fall back to a slower NumPy-based implementation.
>
> **Install compilers:**  
> 
> [![macOS](https://img.shields.io/badge/macOS--lightgrey)](#) `xcode-select --install` 

> [![Linux](https://img.shields.io/badge/Linux-🐧-lightgrey)](#) `sudo apt install build-essential` 

> [![Windows](https://img.shields.io/badge/Windows-💠-lightgrey)](#) *Visual Studio C++ Build Tools*




##### Install from source (development version):

```bash
git clone https://github.com/Charestlab/shinier.git
cd shinier
pip install -e .
```


### 🧩 Demo Example

```python
from shinier import Options, ImageDataset, ImageProcessor, utils

opt = Options(mode=3)  # Luminance matching
dataset = ImageDataset(options=opt)
results = ImageProcessor(dataset=dataset, options=opt, verbose=1)
_ = utils.show_processing_overview(processor=results, img_idx=0)
```

### 💻 **User friendly  Interface**
```bash
shinier --show_results --image_index=1
```

### 💻 **More information**
1. [Package Overview](documentation/documentation.md#overview)
2. [Package Architecture](documentation/documentation.md#package-architecture)
3. [MATLAB vs Python Differences](documentation/documentation.md#matlab-vs-python-differences)
4. [Detailed Processing Modes](documentation/documentation.md#detailed-processing-modes)
5. [Package Main Classes](documentation/documentation.md#main-classes)
6. [Visualization Functions](documentation/documentation.md#visualization-functions)
7. [Implemented Algorithms](documentation/documentation.md#implemented-algorithms)
8. [Memory Management and Performance](documentation/documentation.md#memory-management-and-performance)
9. [Testing and Validation](documentation/documentation.md#testing-and-validation)
