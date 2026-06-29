# API Reference

This page is generated from the Python docstrings. The public narrative
documentation remains in the Markdown files under `documentation/`.

## Core Classes

```{eval-rst}
.. autoclass:: shinier.Options
   :exclude-members: __init__, __new__
   :show-inheritance:

.. autoclass:: shinier.ImageDataset
   :members: close, initialize_dataset, post_init, print_log, save_images
   :exclude-members: __init__, __new__
   :show-inheritance:

.. autoclass:: shinier.ImageProcessor
   :members: get_results, process, lum_match, hist_match, sf_match, spec_match, dithering, ie_methods,    uint8_to_float255, float255_to_float01, float01_to_float255, print_log_results, 
   :exclude-members: __init__, __new__
   :show-inheritance:
```

## Color Processing

```{eval-rst}
.. autoclass:: shinier.color.ColorConverter
   :members: sRGB_to_linRGB, linRGB_to_sRGB, linRGB_to_xyz, xyz_to_linRGB, xyz_to_xyY, xyY_to_xyz, xyz_to_lab, lab_to_xyz, sRGB_to_xyz, xyz_to_sRGB, sRGB_to_lab, lab_to_sRGB, sRGB_to_xyY, xyY_to_sRGB, apply_standard_config
   :exclude-members: __init__, __new__
   :show-inheritance:

.. autoclass:: shinier.color.ColorTreatment
   :members: forward_color_treatment, backward_color_treatment
   :exclude-members: __init__, __new__
   :show-inheritance:

.. autoclass:: shinier.color.GamutControl
   :members: apply_dataset, apply_image, apply_low_Y_desaturation, get_max_luminance_map
   :exclude-members: __init__, __new__
   :show-inheritance:

.. autofunction:: shinier.color.rgb2gray

.. autofunction:: shinier.color.rgb2ntsc_intensity

.. autofunction:: shinier.color.gray2rgb
```

## Utility Functions

### Plotting

```{eval-rst}
.. autofunction:: shinier.utils.hist_plot

.. autofunction:: shinier.utils.imhist_plot

.. autofunction:: shinier.utils.imshow

.. autofunction:: shinier.utils.sf_plot

.. autofunction:: shinier.utils.spectrum_plot

.. autofunction:: shinier.utils.im_power_spectrum_plot

.. autofunction:: shinier.utils.tidhe_hist_plot

.. autofunction:: shinier.utils.show_processing_overview
```

### Utility Classes

```{eval-rst}
.. autoclass:: shinier.utils.Bcolors
   :members:

.. autoclass:: shinier.utils.DiffusionMaps
   :members:

.. autoclass:: shinier.utils.StepSizeController
   :members:

.. autoclass:: shinier.utils.ImageStats
   :members:
```

### StimulusMasker

```{eval-rst}
.. autoclass:: shinier.utils.StimulusMasker
   :members: mask, apply, apply_all, interactive_mask
   :exclude-members: __init__, __new__
```

### MatlabOperators

```{eval-rst}
.. autoclass:: shinier.utils.MatlabOperators
   :members:
```

### Functions

```{eval-rst}
.. autofunction:: shinier.utils.print_shinier_header

.. autofunction:: shinier.utils.pydantic_field_choices

.. autofunction:: shinier.utils.pydantic_model_choices

.. autofunction:: shinier.utils.sf_profile

.. autofunction:: shinier.utils.freq_axis

.. autofunction:: shinier.utils.get_radius_grid

.. autofunction:: shinier.utils.rotational_avg

.. autofunction:: shinier.utils.stretch

.. autofunction:: shinier.utils.convolve_1d

.. autofunction:: shinier.utils.convolve_2d

.. autofunction:: shinier.utils.has_duplicates

.. autofunction:: shinier.utils.n_unique

.. autofunction:: shinier.utils.strict_ordering

.. autofunction:: shinier.utils.exact_histogram

.. autofunction:: shinier.utils.apply_histogram_mapping

.. autofunction:: shinier.utils.floyd_steinberg_dithering

.. autofunction:: shinier.utils.error_diffusion_dither

.. autofunction:: shinier.utils.soft_clip

.. autofunction:: shinier.utils.noisy_bit_dithering

.. autofunction:: shinier.utils.uint_to_float01

.. autofunction:: shinier.utils.float01_to_uint

.. autofunction:: shinier.utils.pol2cart

.. autofunction:: shinier.utils.cart2pol

.. autofunction:: shinier.utils.separate

.. autofunction:: shinier.utils.image_spectrum

.. autofunction:: shinier.utils.gaussian_kernel

.. autofunction:: shinier.utils.center_surround_kernel

.. autofunction:: shinier.utils.laplacian_kernel

.. autofunction:: shinier.utils.tie_breaking_noise_level

.. autofunction:: shinier.utils.print_log

.. autofunction:: shinier.utils.strip_ansi

.. autofunction:: shinier.utils.colorize

.. autofunction:: shinier.utils.console_log

.. autofunction:: shinier.utils.beta_bounds_from_ssim

.. autofunction:: shinier.utils.ssim_sens

.. autofunction:: shinier.utils.hist_match_validation

.. autofunction:: shinier.utils.sf_match_validation

.. autofunction:: shinier.utils.spec_match_validation

.. autofunction:: shinier.utils.compute_rmse

.. autofunction:: shinier.utils.normalized_rmse

.. autofunction:: shinier.utils.get_images_spectra

.. autofunction:: shinier.utils.rescale_image

.. autofunction:: shinier.utils.load_images_from_folder

.. autofunction:: shinier.utils.load_np_array

.. autofunction:: shinier.utils.rescale_images255

.. autofunction:: shinier.utils.uint8_plus

.. autofunction:: shinier.utils.apply_median_blur

.. autofunction:: shinier.utils.hist2list

.. autofunction:: shinier.utils.im3D

.. autofunction:: shinier.utils.imhist

.. autofunction:: shinier.utils.classic_he_gray

.. autofunction:: shinier.utils.tidhe_gray

.. autofunction:: shinier.utils.rdfhe_gray

.. autofunction:: shinier.utils.nfldice_gray

.. autofunction:: shinier.utils.betce_gray

.. autofunction:: shinier.utils.sfcef_gray

```

### Contrast Enhancement Metrics

```{eval-rst}
.. autofunction:: shinier.utils.compute_ambe

.. autofunction:: shinier.utils.compute_contrast_improvement

.. autofunction:: shinier.utils.compute_image_entropy

.. autofunction:: shinier.utils.compute_mssim

.. autofunction:: shinier.utils.compute_psnr

.. autofunction:: shinier.utils.compute_bp2bpsim

```

### Histogram And Image Statistics

```{eval-rst}
.. autofunction:: shinier.utils.rounded_target_hist

.. autofunction:: shinier.utils.compute_tvd_hist

.. autofunction:: shinier.utils.avg_hist

.. autofunction:: shinier.utils.imstats
```

## CLI

```{eval-rst}
.. autofunction:: shinier.SHINIER.prompt

.. autofunction:: shinier.SHINIER.options_display

.. autofunction:: shinier.SHINIER.get_image_list

.. autofunction:: shinier.SHINIER.main
```

## Base Helpers

```{eval-rst}
.. autofunction:: shinier.base.image_list_validator

.. autoclass:: shinier.base.InformativeBaseModel
   :exclude-members: __init__, __new__
```

## Others

### Chroma Loss Metrics

```{eval-rst}
.. autoclass:: shinier.color.chroma_eval.ChromaMetrics

.. autoclass:: shinier.color.chroma_eval.AggregateMetric

.. autoclass:: shinier.color.chroma_eval.AggregateRow

.. autoclass:: shinier.color.chroma_eval.ChromaInfoRetention

.. autoclass:: shinier.color.chroma_eval.ChromaInfoLossResult

.. autofunction:: shinier.color.chroma_eval.chroma_metrics_for_image

.. autofunction:: shinier.color.chroma_eval.aggregate_chroma_metrics

.. autofunction:: shinier.color.chroma_eval.build_chroma_loss_report

.. autofunction:: shinier.color.chroma_eval.chroma_info_loss_bpp

.. autofunction:: shinier.color.chroma_eval.mean_chroma_loss_pct_lab

.. autofunction:: shinier.color.chroma_eval.lab_chroma_retention
```

### ImageListIO

```{eval-rst}
.. autoclass:: shinier.ImageListIO
   :members: readonly_copy, new_copy, copy_with_image_list, to_list, to_gray, get_file_format, final_save_all, close
   :exclude-members: __init__, __new__
   :show-inheritance:
```
