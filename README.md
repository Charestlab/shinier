<p align="center">
  <img src="./assets/SHINIER_logo.jpg" width="500">
</p>


# **Shinier** 🌟 
*A modern Python implementation of the SHINE toolbox with enhancements and new features.*

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Windows%20|%20Linux-informational)](#)  
[![Documentation](https://img.shields.io/badge/docs-coming_soon-orange)](#)  
[![Tests](https://img.shields.io/github/actions/workflow/status/YOUR_GITHUB_USERNAME/shinier/tests.yml?branch=main)](#)

## 🎯 **Overview**
**Shinier** is a Python implementation of the **SHINE** (Spectrum, Histogram, and Intensity Normalization and Equalization) toolbox, originally developed in **MATLAB** [(Willenbockel et al., 2010)](https://doi.org/10.3758/BRM.42.3.671).  
This package introduces **object-oriented programming (OOP)**, performance optimizations, and a range of **new features** while ensuring compatibility with the original methodology.

---

## 🔥 **Key Features**
- **🖼 Dithering** – Improves image quality.
- **📦 Object-Oriented Design** – Modular and extensible.
- **⚡ Optimized Performance** – Efficient memory handling for large dataset.
- **🎨 Color Processing** – New modes for color image analysis.
- **🛡️ Safety Mode** – Avoids out-of-range luminance values.
- **🔢 Higher Number Precision** – Reduces rounding errors in multi-step processing.
- **🕰 Legacy Mode** – Ensures backward compatibility.

---

## 🚀 **Installation**
To install **Shinier**, simply use pip:

```bash
pip install shinier
```
Or install the latest development version:

```bash
pip install git+https://github.com/YOUR_GITHUB_USERNAME/shinier.git
```

**Dependencies:**  
- Python 3.9+
- NumPy (full list in `requirements.txt`)

---

## 🛠️ **Usage**
### **Basic Example**
```python
from shinier import ImageProcessor, Options

# Define processing options
options = Options(safe_lum_match=True, color_mode=True)

# Process an image
processor = ImageProcessor(options)
processed_image = processor.process("input_image.jpg")

# Save the output
processed_image.save("output_image.jpg")
```
### **Batch Processing**
```python
from shinier import ImageDataset

dataset = ImageDataset("input_folder")
dataset.process_all(output_folder="output_folder")
```

---

## 🔧 **Improvements Over MATLAB**
### **🟢 Lazy Loading Mode**
- Reduces RAM usage by loading images **only when necessary**.

### **🎨 Color Mode**
- Adds support for **color image processing** and transformations.

### **🏞️ Legacy Mode**
- Ensures **full compatibility** with previous workflows.

### **🔢 Higher Number Precision**
- For composite modes **5, 6, 7, and 8**, intermediate calculations use **floating-point precision**, reducing rounding errors.

### **🔍 Image Upscaling & Dithering**
- Improves visual quality, reduces banding effects.

---

## 🔍 **Luminance Matching**
### ⚠️ **Possible Out-of-Range Values**
- When images have **different luminance distributions**, some pixels may fall outside **[0, 255]**.  
- In **Shinier**, all images are **clipped before conversion** to `uint8`, preventing unexpected values.

### 🚡 **New Safety Mode**
- When enabled (`safe_lum_match=True`), the algorithm **adjusts parameters automatically**, ensuring values **remain within range**.

---

## 🔬 **Key Differences**
Due to differences in the algorithms implemented by NumPy and MATLAB, we expect the results to be slightly—but sometimes significantly—different. Below are the main differences.  

| **Function** | **Legacy MATLAB Behavior** | **NumPy Behavior**         | **Shinier**                         |
|--------------|----------------------------|----------------------------|-------------------------------------|
| `round`   | Rounds **away from zero**  | Rounds **to nearest even** | Rounds to the nearest even          |
| `uint8`    | Clips values [0,255]       | Truncates & wraps around   | Explicit clipping before conversion |
| `std`     | Uses **N-1** divisor       | Uses **N** (biased)        | Uses N (biased)                     |

📌 *Example Fix for Rounding:*
```python
(np.sign(x) * np.ceil(np.floor(np.abs(x) * 2) / 2)).astype(int)
```

---

## ✅ **Testing & Benchmarking**
All tests were conducted on:
- **Hardware:** MacBook Pro 16-inch (M3 Max, 36GB RAM, 2TB SSD)
- **OS:** macOS 15.3.1
- **Python Environment:** Python 3.9 (see `requirements.txt`)
- **MATLAB Version:** R2024b

### **🤍 Unit Tests**
Tested components:
- **Options** – Parameter handling  
- **ImageListIO** – File input/output  
- **ImageDataset** – Batch processing  
- **ImageProcessor** – Image transformation  

---

## 📚 **Contributing**
We welcome contributions! 🛠️  

### **How to Contribute**
1. **Fork the repository**  
2. **Create a new branch** (`feature/new-feature`)  
3. **Commit changes** (`git commit -m "Added new feature"`)  
4. **Push to your fork** (`git push origin feature/new-feature`)  
5. **Open a Pull Request**  

### **Reporting Issues**
If you find a bug or have a feature request, please open an [issue](https://github.com/YOUR_GITHUB_USERNAME/shinier/issues).

---

## 🐝 **License**
**Shinier** is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🌟 **Acknowledgments**
Special thanks to:
- **Willenbockel et al. (2010)** for the original **SHINE** toolbox.
- The **Python & MATLAB** communities for their contributions to image processing.



# Shinier
This package is a Python implementation of the SHINE toolbox originally coded in MATLAB (see Willenbockel, V., Sadr, J., Fiset, D. et al. Controlling low-level image properties: The SHINE toolbox. Behavior Research Methods 42, 671–684 (2010). https://doi.org/10.3758/BRM.42.3.671). Notable improvements include a new object oriented programming structure, revisions, improvements and various new features. 

## 🔧 Improvements

The legacy code was entirely revised, incorporating major improvements and optimizations. Several new features have been added, enhancing performance, precision, and flexibility. Below are the key new features, along with revisions & improvements for each of the main image processing modes.

---

## 🚀 New Features & Enhancements

#### ⏳ Lazy Loading Mode
Optimizes memory usage by loading data only when needed, reducing RAM consumption and improving performance for large datasets.

#### 🎨 Color Mode
Enables advanced color processing and transformations, allowing greater control over image adjustments and rendering.

#### 🏛️ Legacy Mode
Ensures compatibility with older versions and workflows, preserving previous functionality while integrating new optimizations.

#### 🔢 Higher Number Precision
For composite modes **5, 6, 7, and 8**, which involve two image processing steps, intermediate results are now stored in a **temporary float format** (or **uint16 for discrete processing**). This preserves as much image information as possible, minimizing rounding errors and maintaining high-quality results.

#### 🔍 Image Upscaling Option
After image processing is completed, **dithering** can be applied. This technique improves visual quality by reducing banding effects and increasing smoothness in gradients.


### Luminance matching
#### ⚠️ **Possible out-of-range values** 
When images have very different luminance distributions, luminance matching might end up producing out-of-range values. This is expected when using the average of the mean and the average of the standard deviation of each image as the target distribution parameters. In this package, all images are clipped before converting them to uint8 such that there will be no out-of-range data. 
#### **🛡️ New Safety Mode**

When safety mode is enabled (`safe_lum_match: bool = True`), the algorithm adjusts the target distribution parameters to be as close as possible to the original intended parameters while ensuring that all luminance values remain within the **[0, 255]** range.


# NumPy vs MATLAB: Differences that impact this package

Due to subtle differences in the algorithms implemented by NumPy and MATLAB, we expect the results to be slightly—but sometimes significantly—different. Below are the main differences.

---

## **round.m vs np.round**

MATLAB applies a **round-away-from-zero** algorithm, whereas NumPy applies a **round-half-to-even** algorithm (also known as "Bankers' Rounding").  
This can lead to different results, particularly for numbers exactly halfway between two integers.

#### **MATLAB (`round.m`)**
```matlab
>> round([2.5, 3.5, -2.5, -3.5])
ans =
     3     4    -3    -4
```

#### **Python (`numpy.round`)**
```python
>>> import numpy as np
>>> np.round([2.5, 3.5, -2.5, -3.5])
array([ 2.,  4., -2., -4.])
```
To match MATLAB’s behavior, use the following in NumPy:
```python
(np.sign(x) * np.ceil(np.floor(np.abs(x) * 2) / 2)).astype(int)
```

---

## **uint8.m vs .astype('uint8')**

MATLAB’s `uint8` function rounds numbers to the **nearest integer** and **clips values** between **0 and 255**, whereas NumPy's `.astype('uint8')` applies **truncation and wrap-around behavior** due to the way unsigned integers work in NumPy. In this package, we systematically clip the values between [0, 255] before applying the rounding from NumPy.

#### **MATLAB (`uint8.m`)**
```matlab
>> uint8([2.5, 3.5, -2.5, 255.5])
ans =
     3     4     0   255
```

#### **Python (`np.array.astype`)**
```python
>>> np.array([2.5, 3.5, -2.5, 255.5]).astype('uint8')
array([  2,   3, 254, 255], dtype=uint8)
```

---

## **std2.m vs np.std**

MATLAB’s default standard deviation is **normalized by (N-1) (unbiased estimator)**, whereas NumPy’s default is **normalized by N (biased estimator)**.

#### **MATLAB (`std2.m`)**
```matlab
>> rng(42); % Set seed for reproducibility
>> A = rand(100, 100);
>> format long;
>> std2(A)
ans =
    0.287630126526993
```

#### **Python (`np.std`)**
```python
>>> np.random.seed(42)  # Set seed for reproducibility
>>> A = np.random.rand(100, 100)
>>> np.std(A)  # Default normalization by N
np.float64(0.28761574466111084)
```

To match MATLAB’s behavior (normalization by **N-1**), use `ddof=1` in NumPy:
```python
>>> np.std(A, ddof=1)
np.float64(0.2876301265269928)
```

## Package tests
Al tests were performed on the following:
- Hardware: MacBook Pro 16-inch 2023 (Apple M3 Max), 36GB memory, 2TB SSD 
- OS: Mac OS 15.3.1
- Python environment (see requirements.txt): Python 3.9
- MATLAB: R2024b

-----
### Unit tests
The Options, ImageListIO, ImageDataset and ImageProcessor classes were tested. Placeholder images and masks—generated from noise—were used for all of these tests. 

#### ImageProcessor
All possible processing configurations (i.e. all possible combination of input parameters for the Option class) were tested on a small sample of placeholder noisy images.  

#### How to use ? 
##### Import the necessaries
```
from shinier import ImageDataset, ImageProcessor, Options
```

##### Set the options (see Options.py for more detailed description)
```
options = Options(
    # Paths
    input_folder="testing_INPUT",
    output_folder="testing_OUTPUT",
    masks_folder="testing_MASKS",
    
    # Formats (tif, jpg, png)
    masks_format="png",
    images_format="png",

    # Mode (1 to 8)
    mode = 1,

    # Rescaling options (0 to 2)
    rescaling = 0,

    # What part of the image is processed (1 to 3)
    whole_image=3,

    # RGB or gray scale
    as_gray = False,

    # Save active memory 
    conserve_memory=False,

    # Noise in histogram specification (0 or 1)
    hist_specification = 0,

    # SSIM optimization of histogram matching (0 or 1)
    hist_optim = 1,
    # Number of iterations and step size, (]0, inf])
    iterations = 2,
    step_size=67,

    # Image dithering (True or False)
    dithering= True,

    # Legacy mode (True or False)
    legacy_mode = False,
    
    # Background luminance values in the masks ([0, 255]), !! Must be indicated even if boolean !!
    background=0

    # Random seed, for reproductibility
    seed = 0
    )
```
##### Create the image dataset from the options
```
dataset = ImageDataset(options=options)
```

##### Process the images
```
ImageProcessor(dataset = dataset, verbose = False)
```