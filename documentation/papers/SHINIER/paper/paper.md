---
title: 'The SHINIER the Better: An Adaptation of the SHINE Toolbox on Python'
tags:
  - Python
  - visual perception
  - low-level image properties
  - luminance
  - histogram matching
  - spatial frequency
  - Fourier spectra
authors:
  - name: Mathias Salvas-Hébert
    orcid: 0009-0000-9707-7298
    corresponding: true 
    equal-contrib: true
    affiliation: 1
  - name: Nicolas Dupuis-Roy
    orcid: 0000-0001-9261-0583
    equal-contrib: true 
    affiliation: 2
  - name: Catherine Landry
    orcid: 0000-0001-6748-1417
    affiliation: 1
  - name: Ian Charest
    orcid: 0000-0002-3939-3003
    affiliation: 1
  - name: Frédéric Gosselin
    orcid: 0000-0002-3797-4744
    affiliation: 1
affiliations:
 - name: Département de Psychologie, Université de Montréal, CP 6128, succ. Centre-ville, Montréal, QC, H3C 3J7, CANADA
   index: 1
 - name: Elephant Scientific Consulting, Canada
   index: 2
date: 14 November 2025
bibliography: paper.bib
---

# Summary

SHINIER (Spectrum, Histogram, and Intensity Normalization, Equalization, and Refinement), written in Python, is an open-source package that replicates and extends the functionality of the popular SHINE toolbox [@willenbockel2010controlling], written in MATLAB. Like SHINE, it includes functions for normalizing and scaling mean luminance and contrast, for specifying either the full Fourier amplitude spectrum or its rotational average, and for exact histogram specification. In addition, SHINIER supports color images, better memory management, implements image dithering algorithms for improving pixel depth, and offers improved exact histogram equalization methods, among other enhancements.

# Statement of need

When conducting experiments with humans, animals, or machines, the choice of stimuli is critical. We usually intend observers to rely on features that genuinely support recognition in real life. However, experimental image sets—necessarily small subsets of the virtually infinite possible scenes and viewing conditions—often contain accidental features that can be exploited instead. For example, in a dog–cat categorization task, observers might succeed not because they attend to diagnostic shape or texture cues, but because the dog images (often taken outdoors in bright sunlight) have luminance histograms with higher means and greater variance than cat images (typically photographed indoors under dim lighting). These luminance differences are artifacts of illumination, not reliable distinguishing properties of dogs versus cats in the real world. One way to avoid such confounds is to use artificially generated stimuli with fully controlled low-level properties. Another is to rely on very large naturalistic image databases, such as the Natural Scenes Dataset [@allen2022massive], where idiosyncratic correlations tend to average out. When working with natural images and such large-scale resources are unavailable—or impractical due to time constraints with human or animal participants—normalizing and adjusting low-level image properties, as permitted by the SHINIER package, becomes essential.

# State of the Field

For more than fifteen years, the SHINE (Spectrum, Histogram, and Intensity Normalization and Equalization) toolbox [@willenbockel2010controlling] has served as the primary solution for controlling low-level image properties. It has been cited more than 1,400 times according to Google Scholar—an average of about 100 citations per year—clearly indicating its popularity and utility in the field. The original SHINE toolbox was written in MATLAB for controlling low-level image properties for vision research. Since then, the entire scientific ecosystem has shifted: Python has become the leading programming language of the scientific community [@srinath2017python]; computing capabilities have greatly improved; and vision science practices have evolved. 

To the best of our knowledge, no alternative open-source Python package has achieved comparable adoption or scope in vision science. General-purpose image processing Python libraries such as scikit-image or OpenCV are not designed to support the comprehensive control of low-level image properties required in vision research and addressed by SHINE. In particular, they do not provide exact histogram specification across image sets, spectrum normalization with preserved phase information, or workflows oriented toward reproducible stimulus generation. The SHINIER Python package  (Spectrum, Histogram, and Intensity Normalization, Equalization, and Refinement) fills this gap by reimplementing the full functionality of SHINE in Python, while extending it to support current computational and experimental demands in vision science, including color image processing, high-precision numerical pipelines, and large-scale dataset handling.

# Software Design

## Accessibility

Implementing the SHINIER package in Python enables seamless integration with modern data-science libraries (e.g., pandas, scikit-learn, PyTorch) that are not natively available in MATLAB. However, this choice also introduces legacy numerical compatibility issues stemming from intrinsic differences between Python and MATLAB. For example, the two languages rely on different rounding conventions. Numerical optimizations further increase discrepancies between the MATLAB and Python implementations. SHINIER therefore adopts a flexible numerical design that supports both legacy compatibility—through the legacy_mode option—and updated Python-based numerical handling.

To maximize accessibility while simplifying long-term maintainability across versions, SHINIER was deliberately designed as a lightweight package with minimal dependencies. This design choice, however, required additional effort to preserve low-level performance through optimized implementations (e.g., using Cython for fast convolution), and to integrate external algorithms, such as a dedicated color subpackage for color-space support.

## Simplified Structure

SHINE followed the dominant script-based programming paradigm at the time of its conception. In contrast, SHINIER adopts an object-oriented programming (OOP) architecture that reorganizes the image-processing workflow into three core classes: (1) Options, a user-facing class that defines the input images and associated image processing parameters; (2) ImageDataset, which manages image storage and associated buffers; and (3) ImageProcessor, which applies the processing pipeline in a modular fashion. This modular organization improves maintainability and facilitates extension.

## Improved Numerical Precision and Scalability

Representing the image dataset as an object instantiation of a class provides additional opportunities for optimization and scalability. The numerical precision of images is now tracked across processing stages and, when possible, maintained in 64-bit floating-point format until the final step, where images are converted back to the mandatory 8-bit unsigned integer format. This eliminates precision bottlenecks inherited from the original SHINE workflow.

The use of large datasets whose memory requirements exceed available random-access memory (RAM) is now the norm in computer vision and increasingly common in vision science. To ensure SHINIER’s scalability, we introduced ImageListIO, a dedicated subclass that can handle large datasets. This class leverages fast solid-state drive (SSD) read-and-write operations to circumvent RAM limitations. Specifically, when the conserve_memory option is enabled, images are read from and written on disk on the fly rather than stored in RAM. Although this new approach incurs lower real-time performance, in practice, the overhead is negligible compared to the core image-processing operations of the package. This approach also provides opportunities for future development, such as video processing.

# New Functionalities and Algorithms

SHINIER extends the functionality of SHINE in several ways. The package now supports color images and provides different luminance-processing options. When the linear_luminance option is enabled, input images are assumed to have RGB values linearly related to luminance; color channels are therefore processed independently, matching the original SHINE behavior for legacy compatibility. This approach can, however, produce out-of-gamut or distorted colors. We therefore provide alternative processing options that resolve this issue under different assumptions.

The exact histogram workflow, which enforces identical pixel-intensity distributions across images, was completely revised. Two additional exact histogram specification algorithms were implemented: the method proposed by @coltuc2006exact and a Gaussian-kernel variant. New specification options were also introduced to handle special cases, including images with no identical pixel values—now common with high numerical precision—and situations in which a single algorithm is insufficient. In such cases, a hybrid strategy applies Gaussian ordering first and adds a tiny amount of noise only if ties remain.

Finally, the Floyd–Steinberg dithering algorithm [@floyd1976adaptive] and the noisy-bit dithering algorithm [@allard2008noisy] were added to the package; in both cases, the effective pixel depth of the processed images is increased by exploiting spatial pooling in human and animal visual systems.

# Research Impact Statement

SHINIER should appeal to the hundreds of SHINE users migrating—or planning to migrate—from MATLAB to Python. It should also appeal to new users in the broader scientific community, including vision science and artificial intelligence (AI), who seek to control low-level image properties for applications involving humans, animals, or machines. Support for large image datasets that exceed available RAM, as well as for color images, further extends its applicability. Extensive unit and validation testing ensures numerical accuracy and reproducibility.

The package is openly accessible via “pip install shinier”, with its source code hosted on Charestlab’s GitHub (https://github.com/Charestlab/SHINIER). Comprehensive documentation is provided in the README, and a command-line interface (CLI) allows user-friendly stimulus generation.

# AI Usage Disclosure

Different Large Language Models (LLM)—mainly ChatGPT, Claude, and Gemini—were used to assist in the following tasks: implementing the official style guide for Python code (PEP8); assisting in improving the OOP architecture; writing boilerplate code for classes and functions; providing insights for optimization; writing docstrings and API documentation; writing unit and validation tests. All proposed ideas and code were thoroughly examined, tested, and validated by human developers before committing them into the repository.

# Acknowledgements

The authors would like to thank the original contributors of the SHINE toolbox.

# References