# 🌟 SHINIER
```text
   ███████╗██╗  ██╗██╗███╗  ██╗██╗███████╗██████╗
   ██╔════╝██║  ██║██║████╗ ██║██║██╔════╝██╔══██╗
   ███████╗███████║██║██╔██╗██║██║█████╗  ██████╔╝
   ╚════██║██╔══██║██║██║╚████║██║██╔══╝  ██╔══██╗
   ███████║██║  ██║██║██║ ╚███║██║███████╗██║  ██║
   ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚══╝╚═╝╚══════╝╚═╝  ╚═╝
```
> Spectrum, Histogram, and Intensity Normalization, Equalization, and Refinement.

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/shinier)](https://pypi.org/project/shinier/)
[![PyPI version](https://img.shields.io/pypi/v/shinier.svg)](https://pypi.org/project/shinier/)
[![Tests](https://github.com/Charestlab/shinier/actions/workflows/tests.yml/badge.svg)](https://github.com/Charestlab/shinier/actions/workflows/tests.yml)
---

## 🎯 Overview

SHINIER is a modern Python implementation of SHINE (Spectrum, Histogram, and Intensity Normalization and Equalization), originally developed in MATLAB by Willenbockel et al., 2010. It provides precise control over luminance, contrast, histograms, and spectral content across large image sets for well-calibrated visual experiments.

### Key Features and Improvements

- 🎨 **Color Processing** — New modes for color image control with modern color-space standards (Rec.601 / Rec.709 / Rec.2020).
- 🖼️ **Dithering Support** — Reduces quantization artifacts and enhances output image quality.
- ⚡ **Optimized Performance** — Efficient memory management and faster processing for large image sets (optional Cython/C++ convolution core).
- 🕰 **Legacy Mode** — Ensures full backward compatibility with MATLAB’s original SHINE toolbox.
- 🔢 **High-Precision Arithmetic** — Computations in floating-point precision rather than 8-bit integer space, minimizing rounding errors in multi-stage processing.
- 📦 **Object-Oriented Design** — Modular, extensible architecture with a clean Python API.
- 😀 **User-Friendly CLI** — Guided, prompt-based interface for users who prefer not to write code.

For detailed technical documentation (algorithms, numerical choices, and MATLAB vs Python behavior), see  
[`documentation/documentation.md`](documentation/documentation.md).

---

## 🚀 Quick Start

### Installation

#### Pip Install (recommended):

```bash
pip install shinier
```
> **Note:** SHINIER includes a Cython-compiled C++ extension (`_cconvolve`) for faster convolution. 
> If a C/C++ compiler is available, it will build automatically during installation, otherwise, it will fall back to a slower NumPy-based implementation.
>
> **Install compilers:**
> 
> macOS: `xcode-select --install` 
> 
> Linux: `sudo apt install build-essential` 
> 
> Windows: *Visual Studio C++ Build Tools*

#### Install from source (development version):
```bash
git clone https://github.com/Charestlab/shinier.git
cd shinier
pip install -e ".[dev]"
```

#### Verify the install:
```python
import shinier, sys
print("shinier version:", getattr(shinier, "__version__", "unknown"))
```


### 😀 **User-friendly  Interface**
Call the following bash command to quickly start using the interactive CLI.  
```bash
shinier --show_results --image_index=1
```
<p>
  <img src="https://raw.githubusercontent.com/Charestlab/shinier/main/assets/DEMO_INTERACTIVE_CLI.gif" width="1000" alt="CLI demo">
</p>

### 🧩 Example in Python
Run the following python code to make sure the package is running properly.
```python
from shinier import Options, ImageDataset, ImageProcessor, utils

opt = Options(mode=3)  # Spatial frequency matching
dataset = ImageDataset(options=opt)
results = ImageProcessor(dataset=dataset, options=opt, verbose=1)
_ = utils.show_processing_overview(processor=results, img_idx=0)
```

---
## Processing modes
Change the mode number (e.g. `opt = Options(mode=3)`) to change image processing. See details below:

| Mode | Operations                        | Description                               |
|------|-----------------------------------|-------------------------------------------|
| 1    | `lum_match`                       | Luminance (mean/std) matching             |
| 2    | `hist_match`                      | Histogram matching                        |
| 3    | `sf_match`                        | Rotational spatial frequency matching     |
| 4    | `spec_match`                      | Full 2D Fourier spectrum matching         |
| 5    | `hist_match → sf_match`           | Histogram, then spatial frequency         |
| 6    | `hist_match → spec_match`         | Histogram, then spectrum                  |
| 7    | `sf_match → hist_match`           | Spatial frequency, then histogram         |
| 8    | `spec_match → hist_match` (default) | Spectrum, then histogram (recommended)  |
| 9    | `dithering`                       | Dithering only                            |

Below is an example of results obtained using mode 5 with joint histogram equalization and spatial frequency normalization.
<p>
  <img src="https://raw.githubusercontent.com/Charestlab/shinier/main/assets/demo_cli_mode5.png" width="600" alt="CLI demo">
</p>

---
## 🏛️ **Technical information**
See the accompanying the paper: [The SHINIER the Better: An Adaptation of the SHINE Toolbox on Python](documentation/papers/SHINIER/paper/paper.md)

And documentation:
1. [Package Overview](documentation/documentation.md#overview)
2. [Package Architecture](documentation/documentation.md#package-architecture)
3. [MATLAB vs Python Differences](documentation/documentation.md#matlab-vs-python-differences)
4. [Detailed Processing Modes](documentation/documentation.md#detailed-processing-modes)
5. [Border Artifacts and FFT Padding](documentation/documentation.md#border-artifacts-and-fft-padding)
6. [Package Main Classes](documentation/documentation.md#main-classes)
7. [StimulusMasker](documentation/documentation.md#stimulusmasker)
8. [Visualization Functions](documentation/documentation.md#visualization-functions)
9. [Implemented Algorithms](documentation/documentation.md#implemented-algorithms)
10. [Memory Management and Performance](documentation/documentation.md#memory-management-and-performance)
11. [Testing and Validation](documentation/documentation.md#testing-and-validation)
12. [Usage Demonstrations](documentation/demos.md)
13. [Additional Resources](documentation/documentation.md#additional-resources)

---
## 📚 Citing
If you use **SHINIER**, please cite both of these articles:

### References
- Salvas-Hébert, M., Dupuis-Roy, N., Landry, C., Charest, I., & Gosselin, F. (2025). *The SHINIER the Better: An Adaptation of the SHINE Toolbox on Python*
- Willenbockel, V., Sadr, J., Fiset, D., Horne, G. O., Gosselin, F., & Tanaka, J. W. (2010). Controlling low-level image properties: The SHINE toolbox. *Behavior Research Methods, 42*(3), 671–684. https://doi.org/10.3758/BRM.42.3.671

---
## 🤝 Contributing
See [CONTRIBUTING.md](documentation/contributing.md) for guidelines (coding standards, tests, docs, and PR flow).

---
## 📄 License
See [LICENSE](LICENSE) for more information.

---
## 🛠️ Troubleshooting
- No compiler available: install a C/C++ toolchain or proceed with the NumPy fallback (slower).
- Import errors after upgrade: try pip install --upgrade pip setuptools wheel and reinstall.
- Windows build issues: ensure MSVC Build Tools are installed and on PATH.

---
<p align="center">
  <strong>Code developed by Nicolas Dupuis-Roy and Mathias Salvas-Hébert </strong><br>
  <em>Version 0.2.0 - Complete technical documentation</em>
</p>
