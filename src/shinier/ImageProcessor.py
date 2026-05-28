from __future__ import annotations

import copy
from typing import Optional, List, Union, Iterable, Tuple, Literal, Dict, ClassVar, get_args, Any
from datetime import datetime
from pathlib import Path
import numpy as np
from pydantic import Field, PrivateAttr, ConfigDict

from tqdm import tqdm
import sys

# Local imports
from shinier.base import InformativeBaseModel
from shinier import ImageDataset, Options, ImageListIO
from shinier.utils import (
    beta_bounds_from_ssim, separate, imhist, im3D, cart2pol, pol2cart, soft_clip, imshow,
    rescale_images255, get_images_spectra, ssim_sens, spectrum_plot, imhist_plot, sf_plot, avg_hist,
    uint8_plus, float01_to_uint, uint_to_float01, noisy_bit_dithering, floyd_steinberg_dithering,
    exact_histogram, Bcolors, MatlabOperators, compute_rmse, get_radius_grid, rotational_avg,
    has_duplicates, stretch, console_log, print_log, StepSizeController, image_spectrum, _crop_after_fft
)
from shinier.color import ColorConverter, ColorTreatment, rgb2gray, gray2rgb, RGB2GRAY_WEIGHTS, RGB_STANDARD, GamutControl

RGB_STANDARD_LIST = [r for r in get_args(RGB_STANDARD)]
Vector = Iterable[Union[float, int]]


class ImageProcessor(InformativeBaseModel):
    """Process an image dataset with configurable SHINIER operations.

    This class is designed to process a dataset of images by applying the following
    configurable transformations: luminance matching, histogram matching, spatial frequency
    matching, and Fourier spectrum matching. It is highly customizable and includes
    options like verbosity levels, random seed initialization, and mask generation
    for more fine-grained processing. Validation tests are applied on each transformation
    and all processing steps and important information (e.g. seed) are logged in the output_folder.

    Only ``dataset`` and ``options`` are expected as participant/user inputs.

    Parameters
    ----------
    dataset : ImageDataset
        Dataset containing images, masks, buffers, and processing metadata.

    options : Optional[Options]
        Processing options, including mode selection, verbosity, random seed,
        and matching parameters.

    Runtime Attributes
    ------------------
    - ``bool_masks`` (List): Boolean masks used for masked processing of each image.
    - ``verbose`` (Literal[-1, 0, 1, 2, 3]): Verbosity level controlling console
      output.
    - ``log`` (List[str]): Processing log messages accumulated during execution.
    - ``validation`` (List[dict]): Internal validation results collected after
      processing steps.
    - ``ssim_results`` (List[dict]): Structural Similarity Index (SSIM) test
      results.
    - ``ssim_data`` (List[dict]): SSIM-related intermediate data retained for
      analysis.
    - ``seed`` (Optional[int]): Random seed used for reproducible stochastic
      operations.
    - ``desaturate_chroma_on_low_luminance`` (bool): Whether low-luminance
      chroma desaturation is applied in color treatment.
    - ``from_cli`` (bool): Flag indicating that the processor was launched via
      command-line entry point.
    - ``from_unit_test`` (bool): Internal flag used by unit-test execution paths.
    - ``from_validation_test`` (bool): Internal flag used by validation-test
      execution paths.

    Notes
    -----
    - Uses exact histogram specification without ties whenever possible for
      faster and deterministic results.
    - Input images are transformed into floats [0, 255] and then converted into
      the relevant color space at the beginning of processing. A buffer image
      dataset stores intermediate results. Output images are reconverted back
      into sRGB and uint8 at the end of all image processing steps.
    - All important processing information (e.g. seed) is logged and stored in
      output_folder.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    # --- Public attributes ---
    bool_masks: List = Field(default_factory=list)
    dataset: ImageDataset
    desaturate_chroma_on_low_luminance: bool = Field(default=True)
    from_cli: bool = Field(default=False)
    from_unit_test: bool = Field(default=False)
    from_validation_test: bool = Field(default=False)
    log: List[str] = Field(default_factory=list)
    options: Optional[Options] = None
    seed: Optional[int] = None
    ssim_data: List[dict] = Field(default_factory=list)
    ssim_results: List[dict] = Field(default_factory=list)
    validation: List[dict] = Field(default_factory=list)
    verbose: Literal[-1, 0, 1, 2, 3] = 0

    # --- Private attributes ---
    _backward_conversion_type: str = PrivateAttr(default=None)
    _color_space: Literal['uvw01', 'xyY'] = PrivateAttr(default='xyY')
    _complete: bool = PrivateAttr(default=False)
    _dataset_map: dict = PrivateAttr(default_factory=dict)
    _fct_name2process_name: dict = PrivateAttr(default_factory=dict)
    _final_buffer: Optional[ImageListIO] = PrivateAttr(default=None)
    _forward_conversion_type: str = PrivateAttr(default=None)
    _gamut_control: Optional[GamutControl] = PrivateAttr(default=None)
    _initial_buffer: Optional[ImageListIO] = PrivateAttr(default=None)
    _initial_targets: Optional[Dict[str, np.ndarray]] = PrivateAttr(default={})
    _is_first_operation: bool = PrivateAttr(default=True)
    _is_last_operation: bool = PrivateAttr(default=False)
    _iter_num: int = PrivateAttr(default=0)
    _log_param: dict = PrivateAttr(default_factory=dict)
    _lum_stats: List[np.ndarray] = PrivateAttr(default_factory=list)
    _mode2processing_steps: dict = PrivateAttr(default_factory=dict)
    _n_steps: int = PrivateAttr(default=0)
    _processed_channel: Optional[int] = PrivateAttr(default=None)
    _processed_image: Optional[str] = PrivateAttr(default=None)
    _processing_function: Optional[str] = PrivateAttr(default=None)
    _processing_steps: List[str] = PrivateAttr(default_factory=list)
    _radius_grid: Optional[np.ndarray] = PrivateAttr(default=None)
    _rec_standard: str = PrivateAttr(default="rec709")
    _step: int = PrivateAttr(default=0)
    _sum_bool_masks: List = PrivateAttr(default_factory=list)
    _target_hist: Optional[np.ndarray] = PrivateAttr(default=None)
    _target_lum: Optional[Tuple[Optional[float], Optional[float]]] = PrivateAttr(default=None)
    _target_sf: Optional[np.ndarray] = PrivateAttr(default=None)
    _target_spectrum: Optional[np.ndarray] = PrivateAttr(default=None)

    def post_init(self, __context: Any) -> None:
        """Run initialization logic after Pydantic validation and only once at instantiation."""
        self._forward_conversion_type = f"sRGB_to_{self._color_space}" if self._color_space is not None else None
        self._backward_conversion_type= f"{self._color_space}_to_sRGB" if self._color_space is not None else None

        if self.options is None:
            self.options = getattr(self.dataset, "options", None)

        if not self.bool_masks:
            self.bool_masks = [None] * len(self.dataset.images)
        self.verbose = self.verbose or getattr(self.options, "verbose", 0)

        self._dataset_map = {id(self.dataset.images): "images"}
        if hasattr(self.dataset, "buffer"):
            self._dataset_map[id(self.dataset.buffer)] = "buffer"

        # Initialize GamutControl
        self._gamut_control = GamutControl(
            strategy=self.options.gamut_strategy,
            rec_standard=RGB_STANDARD_LIST[self.options.rec_standard],
            verbose=self.verbose,
            color_space=self._color_space,
        )

        self._mode2processing_steps = {
            1: ["lum_match"],
            2: ["hist_match"],
            3: ["sf_match"],
            4: ["spec_match"],
            5: ["hist_match", "sf_match"],
            6: ["hist_match", "spec_match"],
            7: ["sf_match", "hist_match"],
            8: ["spec_match", "hist_match"],
            9: [None],
        }

        self._fct_name2process_name = {
            "lum_match": "luminance matching",
            "hist_match": "histogram matching",
            "sf_match": "spatial frequency matching",
            "spec_match": "fourier spectrum matching",
            None: "dithering",
        }

        self._rec_standard = RGB_STANDARD_LIST[self.options.rec_standard]
        self._processing_steps = self._mode2processing_steps[self.options.mode]
        self._n_steps = len(self._processing_steps)
        self._sum_bool_masks = [None] * len(self.dataset.images)
        if self.from_unit_test:
            return

        # Run your pipeline
        self.process()
        self.print_log_results()

        if not getattr(self.dataset.images, "has_list_array", False):
            self.dataset.save_images()
            self.dataset.close()
        else:
            console_log(
                msg=(
                    "To get the output images, you must instantiate ImageProcessor "
                    "and call get_results()."
                ),
                indent_level=0,
                color=Bcolors.WARNING,
                verbose=self.verbose >= 2 and not self.from_cli,
            )

    def print_log_results(self):
        """
        Compiles and prints logs related to regular processing steps,
        SSIM validation results, and function validation results. The logs are
        formatted and stored in the specified output folder.
        """

        # Print regular logs
        if self.verbose >= 0:
            logs = []
            if len(self.log) > 0:
                logs = logs + ['[Regular logs]'] + self.log + ['']

            # Print SSIM validation results
            if len(self.ssim_results):
                _logs = [f"iter={res['iter']}; step={res['step']}; image={res['image']}; channel={res['channel']}; result={res['valid_result']}" for res in self.ssim_results]
                logs = logs + ['[SSIM validation results]'] + _logs + ['']

            # Print function validation results
            if len(self.validation):
                _logs = [f"iter={res['iter']}; step={res['step']}; image={res['image']}; channel={res['channel']}; processing function={res['processing_function']}; result={res['valid_result']}; other={res['log_result']}" for res in self.validation]
                logs = logs + ['[Processing validation results]'] + _logs + ['']

            if len(logs):
                print_log(logs=logs, log_path=self.options.output_folder)

    def get_results(self):
        """Return the processed output in the same container style as the input.

        This method is the public accessor for completed runs. It should be called
        after :meth:`process` has finished successfully.

        Returns
        -------
        Union[list[np.ndarray], ImageListIO]
            If the original input images were provided as NumPy arrays, returns the
            processed array data.

            If the input images were loaded from files, returns the same
            ``ImageListIO`` container used by the processor, with the processed
            images stored inside it.

        Raises
        ------
        RuntimeError
            If processing has not been completed yet.
        """
        sp = getattr(self.dataset.images, "store_paths", None)
        if self._complete:
            if not sp or sp[0] is None:
                return self.dataset.images.data
            else:
                return self.dataset.images
        else:
            raise RuntimeError(
                "Cannot retrieve results: ImageProcessor has not run yet. "
                "Call `.process()` before `.get_results()`."
            )

    @staticmethod
    def uint8_to_float255(input_collection: ImageListIO, output_collection: ImageListIO) -> ImageListIO:
        """Convert a uintX collection to float255"""
        for idx, image in enumerate(input_collection):
            output_collection[idx] = image.astype(float)
        output_collection.drange = (0, 255)
        return output_collection

    @staticmethod
    def float255_to_float01(a_list: ImageListIO) -> ImageListIO:
        """Convert a float [0, 255] collection to a float [0, 1]"""
        for idx, image in enumerate(a_list):
            a_list[idx] = image/255
        a_list.drange = (0, 1)
        return a_list

    @staticmethod
    def float01_to_float255(a_list: ImageListIO) -> ImageListIO:
        """Convert a float [0, 1] dataset to a float [0, 255]"""
        for idx, image in enumerate(a_list):
            a_list[idx] = image*255
        a_list.drange = (0, 255)
        return a_list

    def _compute_initial_spectra(self):
        """Compute initial Fourier spectra if required and not already computed."""
        buffer_collection = self.float255_to_float01(self.dataset.buffer)

        # Compute all spectra: Assumes float01
        self.dataset.magnitudes, self.dataset.phases = get_images_spectra(
            images=buffer_collection,
            magnitudes=self.dataset.magnitudes,
            phases=self.dataset.phases,
            rescale=False,
            fft_padding_mode=self.options.fft_padding_mode,
            fft_padding_value=self.options.fft_padding_value)
        self.dataset.buffer = self.float01_to_float255(buffer_collection)

    def _compute_initial_target_spectrum(self):
        """Compute initial target spectrum."""
        target_spectrum = self.options.target_spectrum
        if target_spectrum is None or isinstance(target_spectrum, str):
            target_spectrum = np.zeros(self.dataset.magnitudes[0].shape)
            for idx, mag in enumerate(self.dataset.magnitudes):
                target_spectrum += mag
            target_spectrum /= len(self.dataset.magnitudes)
        elif isinstance(target_spectrum, Path):
            target_spectrum = self._compute_target_spectrum_from_image_path(target_spectrum)
        else:
            target_spectrum = im3D(target_spectrum)
            expected_shape = im3D(self.dataset.magnitudes[0]).shape if self.options.fft_padding_mode != 0 else im3D(self.dataset.buffer[0]).shape
            if target_spectrum.shape != expected_shape:
                if self.options.fft_padding_mode != 0:
                    raise TypeError(
                        "When fft_padding_mode is enabled, a direct target_spectrum array must already "
                        f"match the padded FFT shape. Got {target_spectrum.shape}, expected {expected_shape}."
                    )
                raise TypeError('The target spectrum must have the same size as the images.')
        self._target_spectrum = im3D(target_spectrum)

    def _compute_target_spectrum_from_image_path(self, image_path: Path) -> np.ndarray:
        """Load a target image and compute its spectrum after the same preprocessing as dataset images."""
        load_as_gray = 1 if self.options.as_gray and self.options.linear_luminance else 0
        target_images = ImageListIO(
            input_data=[image_path],
            conserve_memory=True,
            as_gray=load_as_gray,
            save_dir=self.options.output_folder,
        )

        buffer_size = target_images[0].shape[:2] if not self.options.linear_luminance or self.options.as_gray else target_images[0].shape
        target_buffer = ImageListIO(
            input_data=[np.zeros(buffer_size, dtype=bool)],
            conserve_memory=True,
            as_gray=0,
            save_dir=self.options.output_folder,
        )
        target_buffer_other = None
        if not self.options.linear_luminance:
            target_buffer_other = ImageListIO(
                input_data=[np.zeros(buffer_size, dtype=bool)],
                conserve_memory=True,
                as_gray=0,
                save_dir=self.options.output_folder,
            )

        target_buffer = self.uint8_to_float255(target_images, target_buffer)
        target_buffer, _ = ColorTreatment.forward_color_treatment(
            rec_standard=self._rec_standard,
            input_images=target_buffer,
            output_images=target_buffer,
            linear_luminance=self.options.linear_luminance,
            as_gray=self.options.as_gray,
            output_other=target_buffer_other,
            conversion_type=self._forward_conversion_type,
            desaturate_chroma_on_low_luminance=True,
            legacy_mode=self.options.legacy_mode,
            verbose=False,
        )

        if target_buffer.reference_size != self.dataset.buffer.reference_size:
            raise ValueError(
                "The target spectrum image must have the same size as the processed images. "
                f"Got {target_buffer.reference_size}, expected {self.dataset.buffer.reference_size}."
            )

        target_buffer = self.float255_to_float01(target_buffer)
        target_spectrum, _ = image_spectrum(
            target_buffer[0],
            rescale=False,
            fft_padding_mode=self.options.fft_padding_mode,
            fft_padding_value=self.options.fft_padding_value,
        )
        target_images.close()
        target_buffer.close()
        if target_buffer_other is not None:
            target_buffer_other.close()
        return target_spectrum

    def _compute_initial_target_histogram(self, n_bins: int = 256):
        """Compute initial target histogram."""

        # Get target histogram
        target_hist = self.options.target_hist
        if isinstance(target_hist, str) and target_hist in ["equal", "unit_test"]:
            target_hist = np.ones((n_bins, self.dataset.buffer.n_channels))/n_bins
        if target_hist is None:
            target_hist = avg_hist(self.dataset.buffer, binary_masks=self.bool_masks, n_bins=n_bins)
        elif isinstance(target_hist, Path):
            target_hist = self._compute_target_hist_from_image_path(target_hist, n_bins=n_bins)
        else:
            target_hist = target_hist[:, None] if target_hist.ndim == 1 else target_hist
            target_hist /= (target_hist.sum(axis=0, keepdims=True) + 1e-12)
            if target_hist.shape[0] != n_bins:
                raise ValueError(f"target_hist must have {n_bins} bins, but has {target_hist.shape[0]}.")
        if target_hist.ndim > 1 and target_hist.shape[-1] != self.dataset.buffer.n_channels:
            raise ValueError(f"target_hist must have {self.dataset.buffer.n_channels} channels, ")
        self._target_hist = target_hist

    def _compute_target_hist_from_image_path(self, image_path: Path, n_bins: int = 256) -> np.ndarray:
        """Load a target image and compute its histogram after the same preprocessing as dataset images."""
        load_as_gray = 1 if self.options.as_gray and self.options.linear_luminance else 0
        target_images = ImageListIO(
            input_data=[image_path],
            conserve_memory=True,
            as_gray=load_as_gray,
            save_dir=self.options.output_folder,
        )

        buffer_size = target_images[0].shape[:2] if not self.options.linear_luminance or self.options.as_gray else target_images[0].shape
        target_buffer = ImageListIO(
            input_data=[np.zeros(buffer_size, dtype=bool)],
            conserve_memory=True,
            as_gray=0,
            save_dir=self.options.output_folder,
        )
        target_buffer_other = None
        if not self.options.linear_luminance:
            target_buffer_other = ImageListIO(
                input_data=[np.zeros(buffer_size, dtype=bool)],
                conserve_memory=True,
                as_gray=0,
                save_dir=self.options.output_folder,
            )

        target_buffer = self.uint8_to_float255(target_images, target_buffer)
        target_buffer, _ = ColorTreatment.forward_color_treatment(
            rec_standard=self._rec_standard,
            input_images=target_buffer,
            output_images=target_buffer,
            linear_luminance=self.options.linear_luminance,
            as_gray=self.options.as_gray,
            output_other=target_buffer_other,
            conversion_type=self._forward_conversion_type,
            desaturate_chroma_on_low_luminance=True,
            legacy_mode=self.options.legacy_mode,
            verbose=False,
        )

        if target_buffer.reference_size != self.dataset.buffer.reference_size:
            raise ValueError(
                "The target histogram image must have the same size as the processed images. "
                f"Got {target_buffer.reference_size}, expected {self.dataset.buffer.reference_size}."
            )

        target_hist = imhist(target_buffer[0], n_bins=n_bins, normalized=True)
        target_hist = target_hist[:, None] if target_hist.ndim == 1 else target_hist
        target_images.close()
        target_buffer.close()
        if target_buffer_other is not None:
            target_buffer_other.close()
        return target_hist

    def _compute_initial_target_sf(self):
        """Compute initial target rotational average."""
        if self._target_spectrum is None:
            raise ValueError('The target spectrum has not been computed yet.')
        _target_sf = []
        xs, ys, n_channels = self._target_spectrum.shape
        if self._radius_grid is None or self._radius_grid.shape != (xs, ys):
            self._radius_grid = get_radius_grid(x_size=xs, y_size=ys, legacy_mode=self.options.legacy_mode)
        for ch in range(n_channels):
            _target_sf.append(rotational_avg(spectrum=self._target_spectrum[..., ch], radius=self._radius_grid))
        self._target_sf = np.stack(_target_sf).T

    def _get_mask(self, idx):
        """ Provide mask if masks exists in the dataset, if not make blank masks (all True). """
        background = 127 if self.options.whole_image == 3 and self.options.background == 300 else self.options.background
        background_operator = '<' if self.options.whole_image == 3 and self.options.background == 300 else '=='

        def _prepare_mask(image, mask=None):
            if self.options.whole_image == 2:
                self.bool_masks[idx], _, _ = separate(image, background=background, background_operator=background_operator)
            elif self.options.whole_image == 3:
                self.bool_masks[idx], _, _ = separate(mask, background=background, background_operator=background_operator)

        n_dims = self.dataset.buffer.n_dims
        im_size = np.array(list(self.dataset.buffer.reference_size) + [self.dataset.buffer.n_channels])
        if self.bool_masks[idx] is None:
            if self.options.whole_image == 2:
                console_log(msg=f'Preparing mask (whole-image: 2)', indent_level=0, color=Bcolors.HEADER, verbose=self.verbose > 1)
                _prepare_mask(image=self.dataset.images[idx])
            elif self.options.whole_image == 3:
                if idx < self.dataset.n_masks:  # If there is one mask, it picks self.bool_masks[0] everytime
                    console_log(msg=f'Preparing mask (whole-image: 3)', indent_level=0, color=Bcolors.HEADER, verbose=self.verbose > 1)
                    _prepare_mask(image=self.dataset.images[idx], mask=self.dataset.masks[idx])
                else:
                    self.bool_masks[idx] = self.bool_masks[0]
            else:
                # No ROI mask: Whole image is analyzed
                self.bool_masks[idx] = np.ones(im_size, dtype=bool) if idx == 0 else self.bool_masks[0]

            # Resize masks' 3rd dimension
            self.bool_masks[idx] = im3D(self.bool_masks[idx])
            if im_size[2] > self.bool_masks[idx].shape[2]:
                self.bool_masks[idx] = np.repeat(self.bool_masks[idx], 3, axis=2)
            elif im_size[2] < self.bool_masks[idx].shape[1]:
                self.bool_masks[idx] = self.bool_masks[idx][:, :, :im_size[2]]

            # Compute number of unmasked pixels per channel
            self._sum_bool_masks[idx] = [self.bool_masks[idx][..., ch].sum() for ch in range(self.bool_masks[idx].shape[2])]

    def _validate_ssim(self, ssim: List[float]):
        out = np.array(ssim)
        if out.shape[0] > 1:
            for ch in range(out.shape[1]):
                is_strictly_increasing = np.all(np.diff(out[:, ch], axis=0) > -1e-3)
                results = {
                    'iter': self._iter_num,
                    'step': self._step,
                    'image': self._processed_image,
                    'channel': self._processed_channel,
                    'valid_result': is_strictly_increasing
                }
                self.ssim_results.append(results)
                if not is_strictly_increasing and self.verbose > 1:
                    res = f'{Bcolors.OKCYAN}SSIM optimization test for channel {ch}:{Bcolors.ENDC} {Bcolors.FAIL}FAIL{Bcolors.ENDC}'
                    console_log(msg=res, indent_level=1, verbose=self.verbose > 2)
                    raise Exception(f"SSIM optimization non-monotonic for channel {ch}: {out[:, ch]}")

            res = f'{Bcolors.OKCYAN}SSIM optimization test:{Bcolors.ENDC} {Bcolors.OKGREEN}PASS{Bcolors.ENDC}'
            console_log(msg=res, indent_level=1, verbose=self.verbose >= 3)

    def _validate(self, observed: List[float], expected: List[float], measures_str: list[str], rmse_tolerance: float = 1e-3):
        """Internal validation"""
        if len(observed) != len(expected) or len(observed) != len(measures_str):
            raise ValueError('observed, expected and measures_str lists must be the same size')
        diff = [np.abs(obs - expected[idx]) for idx, obs in enumerate(observed)]
        results = {
            'iter': self._iter_num,
            'step': self._step,
            'processing_function': self._processing_function,
            'image': self._processed_image,
            'channel': self._processed_channel,
            'valid_result': np.all([d < rmse_tolerance for idx, d in enumerate(diff)])
        }
        res_color = Bcolors.OKGREEN if results['valid_result'] else Bcolors.FAIL
        res_txt = 'PASS' if results['valid_result'] else 'FAIL'
        res = f'{Bcolors.OKCYAN}Internal test:{Bcolors.ENDC} {res_color}{res_txt}{Bcolors.ENDC}'
        sign = '≮' if res_txt == 'FAIL' else '<'
        obs = '; '.join([f'{msr} (observed vs expected) = |{observed[idx]:4.4f} - {expected[idx]:4.4f}| {sign} {rmse_tolerance:4.4f}' for idx, msr in enumerate(measures_str)])
        if res_txt == 'FAIL' and self.verbose == 3:
            print(res)
            raise Exception(f"At least one difference between expected and observed values is larger than tolerance {rmse_tolerance}: {diff}")
        results['log_result'] = f'{Bcolors.OKBLUE}{obs}{Bcolors.ENDC}\n{res}'
        indent_level = 1 if self._processed_channel is None else 2
        console_log(msg=results['log_result'], indent_level=indent_level, verbose=self.verbose==3)
        self.validation.append(results)

    def process(self):
        """Run the full SHINIER pipeline on the current dataset.

        This method performs the complete processing workflow selected by
        ``options.mode``. Input images are first copied into the working buffer and
        converted to float values in ``[0, 255]``. Depending on the selected mode,
        the processor may then apply luminance matching, histogram matching,
        spatial-frequency matching, Fourier-spectrum matching, or dithering only.

        The processing order and number of iterations are controlled by
        ``options.mode`` and ``options.iterations``. When histogram-based methods are
        used, a random seed is created if needed and recorded in the log so that the
        run can be reproduced. Color conversion behavior is driven by
        ``options.as_gray``, ``options.linear_luminance``, ``options.rec_standard``,
        and ``options.gamut_strategy``. Final conversion to ``uint8`` is handled by
        ``options.dithering`` and can use noisy-bit or Floyd-Steinberg dithering.

        Notes
        -----
            - ``legacy_mode`` switches several sub-steps to MATLAB-compatible
                behavior.
            - ``conserve_memory`` and the input storage mode affect how much data is
                kept in memory during processing.
            - Fourier-based modes reuse the padding configuration from
                ``options.fft_padding_mode`` and ``options.fft_padding_value``.
        """

        # Set a seed for the random generator used in exact histogram specification
        if self.seed is None:
            now = datetime.now()
            self.seed = int(now.timestamp())
        np.random.seed(self.seed)
        self.log.append(f'seed={self.seed}')
        console_log(msg=f'\n\nUse this seed for reproducibility: {self.seed}', color=Bcolors.WARNING, indent_level=0, verbose=self.verbose>=1 and not self.from_cli)

        # Put input images into buffer dataset and convert to float [0, 255]
        self.dataset.buffer = self.uint8_to_float255(self.dataset.images, self.dataset.buffer)

        # Apply relevant color treatment
        if self.options.as_gray == False or self.options.linear_luminance == False:
            msg = f'\nApplying pre-processing color treatment'
            if self.desaturate_chroma_on_low_luminance:
                msg = msg + f' and desaturation on low luminance (Y < {ColorTreatment.Y_desaturation_threshold}) chroma'
            msg = msg + '...'
            console_log(
                msg=msg,
                indent_level=0,
                color=Bcolors.SECTION,
                verbose=self.verbose >= 1
            )
        self.dataset.buffer, self.dataset.buffer_other = ColorTreatment.forward_color_treatment(
            rec_standard=self._rec_standard,
            input_images=self.dataset.buffer,
            output_images=self.dataset.buffer,
            linear_luminance=self.options.linear_luminance,
            as_gray=self.options.as_gray,
            output_other=self.dataset.buffer_other,
            conversion_type=self._forward_conversion_type,
            desaturate_chroma_on_low_luminance=True,
            legacy_mode=self.options.legacy_mode,
            verbose=self.verbose>=2)

        # Copy of the original buffer
        self._initial_buffer = self.dataset.buffer.new_copy(to_list=False)

        # Compute target histogram if required
        if self.options.mode in [2, 5, 6, 7, 8]:
            # Prepare masks
            if self.options.mode == 2:
                for idx in range(self.dataset.images.n_images):
                    self._get_mask(idx)
            self._compute_initial_target_histogram()
            self._initial_targets['hist'] = self._target_hist.copy()


        # Compute Fourier spectra and target spectrum if required
        if self.options.mode in [3, 4, 5, 6, 7, 8]:
            self._compute_initial_spectra()
            self._compute_initial_target_spectrum()
            self._initial_targets['spectrum'] = self._target_spectrum.copy()

        # Compute target sf if required
        if self.options.mode in [3, 5, 7]:
            self._compute_initial_target_sf()
            self._initial_targets['sf'] = self._target_sf.copy()

        # Set tqdm
        if self.verbose == 0:
            total_steps = self.options.iterations * len(self._processing_steps)
            pbar = tqdm(
                total=total_steps,
                unit="composite iteration",
                ncols=80,
                dynamic_ncols=True,
                colour="green",
                file=sys.stdout
            )

        # The outter loop runs n times, one time per processing steps associated with given mode.
        # The inner loop is for modes which have multiple steps (e.g. multiple iterations for optimization)
        mask_prepared = False
        cnt = 0
        for self._iter_num in range(self.options.iterations):
            for self._step, self._processing_function in enumerate(self._processing_steps):
                self._is_last_operation = self._iter_num == self.options.iterations - 1 and self._step == self._n_steps - 1
                if self._processing_function is not None:
                    # Prepare masks if not done
                    if not mask_prepared:
                        for idx in range(self.dataset.images.n_images):
                            self._get_mask(idx)
                            mask_prepared = True

                    # Get the processing function, check and call it
                    if self.verbose == 0:
                        pbar.set_description(f'{Bcolors.BOLD}Applying {self._fct_name2process_name[self._processing_function]}... (iter={self._iter_num}, step={self._step}){Bcolors.ENDC}')

                    exec_fct = getattr(self, self._processing_function, None)
                    if exec_fct is None:
                        raise RuntimeError(f'Function {self._processing_function} does not exist in ImageProcessor class')

                    console_log(
                        msg=f'\nApplying {self._fct_name2process_name[self._processing_function]}... (iter={self._iter_num}, step={self._step})',
                        indent_level=0,
                        color=Bcolors.SECTION,
                        verbose=self.verbose >= 1
                    )
                    exec_fct()
                    print('') if self.verbose > 1 else None  # Adds \n between process
                    if self.verbose == 0:
                        pbar.update(1)
                        cnt += 1

                self._is_first_operation = False

        # Copy of the final buffer (before
        self._final_buffer = self.dataset.buffer.new_copy(to_list=False)

        # Apply relevant global gamut control strategy
        if self.dataset.buffer_other is not None and self.options.gamut_strategy in ['constrain_dataset_luminance', 'constrain_dataset_chrominance']:
            console_log(f'Applying gamut control strategy: {self.options.gamut_strategy}', indent_level=0, verbose=self.verbose >= 1, color=Bcolors.SECTION)
            self.dataset.buffer, self.dataset.buffer_other = self._gamut_control.apply_dataset(self.dataset.buffer, self.dataset.buffer_other, verbose=self.verbose >= 1)

        # Apply relevant inverse color treatment
        if self.options.as_gray == False or self.options.linear_luminance == False:
            console_log(
                msg=f'\n\nApplying post-processing color treatment...',
                indent_level=0,
                color=Bcolors.SECTION,
                verbose=self.verbose >= 1)

        self.dataset.buffer = ColorTreatment.backward_color_treatment(
            rec_standard=self._rec_standard,
            input_images=self.dataset.buffer,
            input_other=self.dataset.buffer_other,
            output_images=self.dataset.buffer,
            linear_luminance=self.options.linear_luminance,
            gamut_strategy=self.options.gamut_strategy,
            as_gray=self.options.as_gray,
            conversion_type=self._backward_conversion_type,
            verbose=self.verbose>=2)

        # Applies dithering or simply convert into uint8 if no dithering
        self.dataset.images = self.dithering(
            input_collection=self.dataset.buffer,
            output_collection=self.dataset.images,
            dithering=self.options.dithering)
        self._complete = True

    def dithering(self, input_collection: ImageListIO, output_collection: ImageListIO, dithering: Literal[0, 1, 2]):
        """Convert the processed float images back to ``uint8`` with optional dithering.

        This function processes each image in the input collection individually and applies the appropriate
        dithering method. Supported methods include noisy bit dithering, Floyd-Steinberg dithering, and
        a standard uint8 conversion, depending on the specified mode. The processed images are stored
        in the provided output collection.

        Parameters
        ----------
        input_collection : ImageListIO
            Buffer of processed images in float form, typically in ``[0, 255]``.
        output_collection : ImageListIO
            Destination collection that receives the converted images.
        dithering : Literal[0, 1, 2]
            Dithering mode from ``options.dithering``.
            - 0 = disables dithering
            - 1 = applies noisy-bit dithering
            - 2 = applies Floyd-Steinberg dithering

        Returns
        -------
        ImageListIO
            Collection containing the converted images (or unchanged conversion
            results when dithering is disabled).
        """
        # Dithering function assumes float with values in the [0, 1] range.
        for idx, image in enumerate(input_collection):
            if dithering == 1:  # Make sure images are float01
                output_collection[idx] = noisy_bit_dithering(image=image/255, depth=256, legacy_mode=self.options.legacy_mode)
            elif dithering == 2:
                output_collection[idx] = floyd_steinberg_dithering(image=image/255, depth=256, legacy_mode=self.options.legacy_mode)
            else:
                output_collection[idx] = MatlabOperators.uint8(image) if self.options.legacy_mode else uint8_plus(image=image, verbose=self.verbose==3)

        return output_collection

    def lum_match(self):
        """Match the luminance statistics of each image to a target.

        This method adjusts the mean and/or standard deviation of each masked image
        region. The target values come from ``options.target_lum``:

        - ``target_lum[0]`` controls the mean.
        - ``target_lum[1]`` controls the standard deviation.

        A value of ``0`` means "use the dataset average" for that statistic, while
        ``None`` leaves the statistic unchanged. When ``safe_lum_match`` is enabled,
        the method searches for nearby statistics that avoid values outside
        ``[0, 255]``.

        Notes
        -----
        - The operation works on the processed buffer after the color treatment
          selected by ``options.as_gray`` and ``options.linear_luminance``.
        - Clipping should happen before ``uint8`` conversion because NumPy wraps
          out-of-range values when casting to ``uint8``.
        - In grayscale mode, the matching is applied directly to the intensity
          values.
        """

        def predict_values(original_means, original_stds, original_min_max, target_mean, target_std):
            """
            Predicts the minimum, maximum, and range values for a target distribution by transforming the
            original minimum and maximum values based on the provided target mean and standard deviation.
            The transformation is performed using the Z-score normalization formula.
            """
            target_mean = np.asarray(target_mean, dtype=float)
            target_std = np.asarray(target_std, dtype=float)
            predicted_min = (original_min_max[:, 0] - original_means) / original_stds * target_std + target_mean
            predicted_max = (original_min_max[:, 1] - original_means) / original_stds * target_std + target_mean
            predicted_range = predicted_max - predicted_min
            return predicted_min, predicted_max, predicted_range

        def compute_stats(im: np.ndarray, binary_mask: np.ndarray) -> Tuple[float, float, float, float]:
            """M, SD, min, max from RGB images whose channels are weighted (or not if grayscale).

            Parameters
            ----------
            im : np.ndarray
                An image.
            binary_mask : np.ndarray
                A mask.

            Returns
            -------
            Tuple[float, float, float, float]
                Mean, standard deviation, minimum, and maximum.
            """
            im = im3D(im)
            binary_mask = im3D(binary_mask)
            M = MatlabOperators.mean2(im[binary_mask]) if self.options.legacy_mode else np.mean(im[binary_mask])
            SD = MatlabOperators.std2(im[binary_mask]) if self.options.legacy_mode else np.std(im[binary_mask])
            min = np.min(im[binary_mask])
            max = np.max(im[binary_mask])
            # if self.options.as_gray != 0:
            #     M = MatlabOperators.mean2(im[binary_mask]) if self.options.legacy_mode else np.mean(im[binary_mask])
            #     SD = MatlabOperators.mean2(im[binary_mask]) if self.options.legacy_mode else np.mean(im[binary_mask])
            # else:
            #     convertion_type = RGB2GRAY_WEIGHTS['int2key'][self.options.rgb_weights]
            #     ch_weights = RGB2GRAY_WEIGHTS[conversion_type]
            #     ch_means = np.array([np.mean(im[:, :, c][binary_mask[:, :, c]]) for c in range(3)])
            #     ch_stds = np.array([np.std(im[:, :, c][binary_mask[:, :, c]]) for c in range(3)])
            #     M = np.sum(ch_means * ch_weights)
            #     SD = np.sqrt(np.sum((ch_weights ** 2) * (ch_stds ** 2)))

            return M, SD, min, max

        def resolve_target(value, originals):
            """Map None to original values, 0 to dataset average, and numbers to themselves."""
            return originals if value is None else float(np.mean(originals)) if value == 0 else float(value)

        def as_target_array(value, n_images):
            """Ensures 1 target value per image."""
            return np.full(n_images, value, dtype=float) if np.ndim(value) == 0 else np.asarray(value, dtype=float)

        buffer_collection = self.dataset.buffer

        # 1) Compute the mean and standard deviation of the original images.
        # 2) Resolve the target mean and standard deviation.
        # 3) Adjust the target mean and standard deviation if safe_values is enabled and if there are out-of-range values.
        # 4) Convert images into float
        # 5) Rescale according to target mean and standard deviation
        # 6) Apply clipping if needed
        # 7) Convert images back into uint8
        target_lum = self.options.target_lum
        safe_values = self.options.safe_lum_match
        original_means, original_stds, original_min_max = [], [], []
        self._processed_channel = None
        for idx, im in enumerate(buffer_collection):
            M, SD, min, max = compute_stats(im=im, binary_mask=self.bool_masks[idx])
            original_means.append(M)
            original_stds.append(SD)
            original_min_max.append((min, max))

        original_means, original_stds = np.array(original_means, dtype=float), np.array(original_stds, dtype=float)
        partial_mean,   partial_std   = target_lum[0] is None, target_lum[1] is None
        target_mean,    target_std    = resolve_target(target_lum[0], original_means), resolve_target(target_lum[1], original_stds)

        original_min_max = np.array(original_min_max, dtype=float)
        predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)

        if safe_values and (any(predicted_min < 0) or any(predicted_max > 255)):
            if partial_std:
                raise ValueError(
                    "safe_lum_match cannot keep values within [0, 255] for target_lum=(mean, None) "
                    "without changing the original contrasts."
                )
            if partial_mean:
                safe_std_limits = []
                for original_mean, original_std, (original_min, original_max) in zip(original_means, original_stds, original_min_max):
                    # Find largest std that keeps values within [0, 255] for this image, given the target mean.
                    if original_min < original_mean:
                        safe_std_limits.append(original_mean * original_std / (original_mean - original_min))
                    if original_max > original_mean:
                        safe_std_limits.append((255 - original_mean) * original_std / (original_max - original_mean))
                # Use most constraining std limit across images to ensure all values are within [0, 255]
                max_safe_std = min(safe_std_limits)
                scaling_factor = min(1, (max_safe_std - 1e-6) / target_std)
            else:
                max_range = predicted_max.max() - predicted_min.min()
                scaling_factor = np.min([1, (255 - 1e-6) / max_range])  # Safety margin of 1e-6 to avoid precision issues
            target_std *= scaling_factor
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            if not partial_mean:
                target_mean = target_mean + (255 - np.max(predicted_max))
                predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            target_mean_log = np.mean(target_mean) if np.ndim(target_mean) else target_mean
            target_std_log = np.mean(target_std) if np.ndim(target_std) else target_std
            console_log(msg=f"Adjusted target values for safe values: M = {target_mean_log:.4f}, SD = {target_std_log:.4f}", indent_level=0,color=Bcolors.WARNING, verbose=self.verbose > 2)
            if np.any(predicted_min < -1e-3) or np.any(predicted_max > (255 + 1e-3)):
                raise Exception(f'Out-of-range values detected: mins = {list(predicted_min)}, maxs = {list(predicted_max)}')
        # 1 value per image
        target_means = as_target_array(target_mean, len(original_means))
        target_stds = as_target_array(target_std, len(original_stds))
        self._target_lum = (
            None if partial_mean else float(target_means[0]),
            None if partial_std else float(target_stds[0]),
        )
        for idx, im in enumerate(buffer_collection):
            im2 = im3D(im.copy())
            M, SD, min, max = compute_stats(im=im2, binary_mask=self.bool_masks[idx])
            target_mean_i = target_means[idx]
            target_std_i = target_stds[idx]

            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            console_log(msg=f"\nImage {self._processed_image}", indent_level=0, color=Bcolors.BOLD, verbose=self.verbose>=2)
            console_log(msg=f"Original: M = {M:.4f}, SD = {SD:.4f}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose==3)

            # Standardization
            if original_stds[idx] != 0:
                im2[self.bool_masks[idx]] = (im2[self.bool_masks[idx]] - original_means[idx]) / original_stds[idx] * target_std_i + target_mean_i
            else:
                im2[self.bool_masks[idx]] = target_mean_i

            M, SD, min, max = compute_stats(im=im2, binary_mask=self.bool_masks[idx])

            # Save resulting image
            console_log(msg=f"Target values: M = {target_mean_i:.4f}, SD = {target_std_i:.4f}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose==3)
            self.dataset.buffer[idx] = im2  # update the dataset

            checks = []
            if not partial_mean:
                checks.append((M, target_mean_i, 'M'))
            if not partial_std:
                checks.append((SD, target_std_i, 'SD'))
            observed, expected, measures = map(list, zip(*checks))
            self._validate(observed=observed, expected=expected, measures_str=measures)

        self.dataset.buffer.drange = (0, 255)

    def hist_match(self):
        """Match each image histogram to the target histogram.

        The target histogram is taken from ``options.target_hist`` and is matched
        with an exact histogram specification routine on 8-bit bins
        (``n_bins = 256``).

        Tie handling is selected by ``options.hist_specification``:

        - 1 = ``noise``
        - 2 = ``moving-average``
        - 3 = ``gaussian``
        - 4 = ``hybrid``

        In practice, this implementation first checks whether ties are present
        in the current image. If no ties are detected, it forces the fast
        deterministic mapping path (``tie_strategy='none'``).

        If ``options.hist_optim`` is enabled, the method performs additional SSIM
        guided sub-iterations using sensitivity maps and adaptive step-size
        control. The number of optimization iterations comes from
        ``options.hist_iterations`` (implemented as ``hist_iterations + 1`` so the
        last pass remains an exact histogram projection). The random seed set in
        ``process`` is reused for reproducibility when tie-breaking requires
        stochastic behavior.

        Notes
        -----
        - Matching is performed inside the current masks when masks are active.
        - Behavior depends on color-treatment mode. In the default color path
          (``linear_luminance=False`` and ``as_gray=False``), histogram matching
          is applied to the luminance channel only and chromaticity is restored
          later. With ``linear_luminance=True`` and ``as_gray=False``, matching is
          done independently per RGB channel. With ``as_gray=True``, matching is
          done on grayscale intensities.
        - Output is kept as float ``[0, 255]`` in the working buffer.
        - In optimization mode, SSIM trends are monitored and can trigger
          rollback/stall stopping through ``StepSizeController``.

        Raises
        ------
        ValueError
            If ``target_hist`` is incompatible with the processed images.
            Typical cases are:
            - wrong number of bins (must be 256),
            - wrong number of channels for the active processing mode,
            - target histogram image path with a size different from the
              processed dataset reference size.
        """

        hist_optim = self.options.hist_optim
        hist_spec_names = ['noise', 'moving-average', 'gaussian', 'hybrid', 'none']
        hist_specification = 5 if self.options.hist_specification is None else self.options.hist_specification
        tie_strategy = hist_spec_names[hist_specification-1]

        # Get appropriate image collection
        buffer_collection = self.dataset.buffer
        bit_size = 8
        n_bins = 2 ** bit_size

        if not self._is_first_operation and self.options._is_moving_target and self.options.mode > 2:
            self._compute_initial_target_histogram()

        # If hist_optim disable, will run only one loop (n_iter = 1)
        n_iter = self.options.hist_iterations + 1 if hist_optim else 1  # See important note below to explain the +1. Also, note that the number of iterations for SSIM optimization (default = 10)
        step_sizes = np.zeros(buffer_collection.n_channels)  # Step size (default = 34)

        # Match the histogram
        self._processed_channel = None
        for idx, image in enumerate(buffer_collection):
            original_image = self._initial_buffer[idx] if self.options._is_moving_target else image
            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            console_log(msg=f"\nImage {self._processed_image}", indent_level=0, color=Bcolors.BOLD, verbose=self.verbose>=2)

            step_sizes_weight = .5
            controller = StepSizeController(gain_up=1.3, gain_down=0.6) if hist_optim else None
            ssim_prev = 0.0
            all_ssim = []
            ssim = None
            ssim_increment = []
            image = im3D(image)
            X = image.copy()
            self._sub_iter = 0
            while self._sub_iter < n_iter:  # n_iter = 1 when hist_optim == False
                if n_iter > 1 and self._sub_iter < n_iter - 1:
                    console_log(msg=f"\nSSIM optimization (iter={self._sub_iter + 1})", indent_level=1, color=Bcolors.BOLD, verbose=self.verbose >= 2)

                # Compute histogram specification
                has_isoluminant_pixels = has_duplicates(X, binary_mask=self.bool_masks[idx])
                if not has_isoluminant_pixels or (tie_strategy == 'none' and not (self._is_last_operation and self._sub_iter == (n_iter - 1))):
                    if not has_isoluminant_pixels and tie_strategy != 'none':
                        console_log(msg=f"No ties detected: using direct histogram mapping", indent_level=1, color=Bcolors.OKCYAN, verbose=self.verbose == 3)
                    Y, _ = exact_histogram(image=X, binary_mask=self.bool_masks[idx], target_hist=self._target_hist, tie_strategy='none', n_bins=n_bins)
                else:
                    Y, OA = exact_histogram(image=X, binary_mask=self.bool_masks[idx], target_hist=self._target_hist, tie_strategy=tie_strategy, n_bins=n_bins)
                    if hist_spec_names != 'noise' and (n_iter == 1 or (n_iter > 1 and self._sub_iter < n_iter - 1)):
                        console_log(msg=f"Ordering accuracy per channel = {OA}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose == 3)
                # Compute Structural Similarity and gradient map (sens), along with max and min
                if self._sub_iter < n_iter - 1:
                    sens, ssim = ssim_sens(original_image, Y, data_range=n_bins-1, use_sample_covariance=False, binary_mask=self.bool_masks[idx])

                if self._sub_iter == n_iter - 1:
                    break

                # Compute theoretical bounds for step size on first iteration only
                if self._sub_iter == 0:
                    beta_bounds = beta_bounds_from_ssim(gradients=sens, ssim=ssim, binary_mask=self.bool_masks[idx])
                    for ch in range(len(beta_bounds)):
                        min_beta, max_beta = beta_bounds[ch][0], beta_bounds[ch][1]
                        step_sizes[ch] = (max_beta - 0)

                if n_iter > 1 and self._sub_iter < n_iter - 1:
                    all_ssim.append(ssim)
                    ssim_increment.append(np.mean(ssim))

                if hist_optim and (n_iter == 1 or (n_iter > 1 and self._sub_iter < n_iter - 1)):
                    console_log(msg=f"Mean SSIM = {np.mean(ssim):.5f}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose==3)

                # Update step size weight:
                #   - if SSIM decreases: rollback one iteration with lower weight
                #   - if SSIM stalls: rollback one iteration with higher weight
                #   - if SSIM increases: continue
                restart, done = False, False
                if hist_optim and self._sub_iter < n_iter - 1:
                    ssim_mean = float(np.mean(ssim))
                    previous_weight = step_sizes_weight
                    step_sizes_weight, ssim_mean, Y, sens, restart, done = controller.update(step_sizes_weight, ssim_mean, Y, sens)
                    if done:
                        console_log(f'Exiting optimization: SSIM stalled for more than {controller.stall_iter - 1} iterations (see StepSizeController.max_stall_iter)', indent_level=1, color=Bcolors.WARNING, verbose=self.verbose>=3)
                        break
                    if 'rollback' in controller.restart_reason:
                        console_log(f"{controller.restart_reason}: step_size_weight= {previous_weight: 1.5f} -> {step_sizes_weight: 1.5f}", indent_level=1, color=Bcolors.WARNING, verbose=self.verbose>=3)

                # Update image with gradient maps, step size and weight
                if hist_optim and self._sub_iter < n_iter - 1:
                    mask_sum = np.array(self._sum_bool_masks[idx].copy())
                    X = Y + sens * step_sizes[np.newaxis, np.newaxis, :] * mask_sum[np.newaxis, np.newaxis, :] * step_sizes_weight
                    if restart:
                        # restart from last good point
                        continue

                self._sub_iter += 1

            # Test monotonic increase of ssim between first and last iteration
            if self.options.hist_optim and len(all_ssim) >=2:
                self._validate_ssim(ssim=[all_ssim[0], all_ssim[-1]])

            # Important Note:
            # - Must use Y as this is the one that matches the target histogram.
            # - n_iter is adjusted to +1 to assure the proper number of optimization steps is run.
            # new_image = np.rint(np.clip(Y, 0, 2 ** bit_size - 1)).astype(f'uint{bit_size}')
            new_image = Y.copy()

            # Make sure the output is always float255
            buffer_collection[idx] = Y / (n_bins - 1) * 255

            # Compute statistics
            final_hist = imhist(image=new_image, mask=self.bool_masks[idx], n_bins=n_bins, normalized=True)
            rmse = compute_rmse(final_hist.flatten(), self._target_hist.flatten())
            if ssim is not None:
                console_log(msg=f"SSIM index between transformed and original image: {np.mean(ssim):.5f}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose==3)
            self._validate(observed=[rmse], expected=[0], measures_str=['RMS error'])

        buffer_collection.drange = (0, 255)

    def sf_match(self):
        """Match the rotationally averaged spatial-frequency content of each image 
        (i.e., mean amplitude per spatial frequency).

        The target frequency profile is derived from ``options.target_spectrum``
        and the initial padded spectra computed through the Fourier pipeline. The
        method scales each channel's magnitude spectrum to match the target radial
        average while preserving the original phase.

        Notes
        -----
        - Spectra are computed upstream from float ``[0, 1]`` images. If FFT
          padding is enabled, it is controlled by ``options.fft_padding_mode`` and
          ``options.fft_padding_value``.
        - The target profile is recomputed when the pipeline uses moving targets
          (composite iterative modes).
        - Frequencies above the Nyquist limit are explicitly zeroed to avoid
          aliasing artifacts.
        - The implementation switches between Cartesian and polar Fourier
          representations (``cart2pol``/``pol2cart``) so magnitude can be
          reweighted while phase is preserved.
        - Channel phases are preserved; only magnitude is reweighted to match the
          target rotational average.
        - The adjustment is done independently per channel.
        - Intermediate outputs can go outside ``[0, 1]``. This is expected in
          iterative modes ``5`` and ``7`` and is readjusted when
          ``self._is_last_operation`` through soft clipping and optional
          rescaling.
        """

        if not self._is_first_operation and self.options._is_moving_target and self.options.mode > 3:
            self._compute_initial_spectra()
            self._compute_initial_target_spectrum()
            self._compute_initial_target_sf()
        buffer_collection = self.dataset.buffer

        # Compute Nyquist and radius grid
        x_size, y_size, n_channels = self._target_spectrum.shape[:3]
        nyquistLimit = np.floor(min(x_size, y_size) / 2)
        r_int = self._radius_grid

        # Match spatial frequency on rotational average of the magnitude spectrum
        for idx, image in enumerate(buffer_collection):
            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            console_log(msg=f"\nImage {self._processed_image}", indent_level=0, color=Bcolors.BOLD, verbose=self.verbose>=2)
            matched_image = []
            original_shape = image.shape[:2]
            magnitude = im3D(self.dataset.magnitudes[idx])
            phase = im3D(self.dataset.phases[idx])
            for self._processed_channel in range(n_channels):
                console_log(msg=f'\nChannel {self._processed_channel}', indent_level=1, verbose=self.verbose >= 2, color=Bcolors.BOLD)
                fft_image = magnitude[:, :, self._processed_channel]

                # Rotational averages (target vs source) as MEANS over annuli
                target_ra = self._target_sf[:, self._processed_channel].squeeze()
                source_ra = rotational_avg(spectrum=fft_image, radius=r_int)

                # Per-radius scale coefficients; avoid divide-by-zero on empty/zero annuli
                coef = target_ra / np.maximum(source_ra, 1e-12)

                # For where in r the value is j, apply the coefficient of index j to cmat
                cmat = coef[r_int]

                # Remove frequencies higher than the Nyquist frequency
                cmat[r_int > nyquistLimit] = 0  # zero beyond Nyquist

                # Compute new magnitude and convert back to image
                new_magnitude = fft_image * cmat

                XX, YY = pol2cart(new_magnitude, phase[:, :, self._processed_channel])
                new = XX + YY * 1j  # 1j = sqrt(-1)

                output = np.real(np.fft.ifft2(np.fft.ifftshift(new)))
                matched_image.append(_crop_after_fft(output, original_shape))

                # Comparison: obtained vs target rotational averages (up to Nyquist)
                obtained_ra = rotational_avg(spectrum=new_magnitude, radius=r_int)
                R = int(min(len(obtained_ra), len(target_ra), nyquistLimit + 1))
                t = target_ra[:R]
                o = obtained_ra[:R]
                rmse = compute_rmse(t, o)
                self._validate(observed=[rmse], expected=[0], measures_str=['RMS error'])

            # Soft-clip output values: As this transformation typically produces out-of-range values
            output_image = np.stack(matched_image, axis=-1).squeeze()
            if self._is_last_operation:
                mn, mx = output_image.min(), output_image.max()
                if mn < 0 or mx > 1:
                    console_log(
                        msg=f'Out of range values: Actual range [{mn}, {mx}] outside of the admitted range [0, 1].\nWill be rescaled and clipped so that less than 1% falls outside of [0, 1].',
                        indent_level=1, color=Bcolors.WARNING, verbose=self.verbose==3)
                    output_image = soft_clip(output_image, min_value=0, max_value=1, max_percent=0.01, verbose=self.verbose==3)
            buffer_collection[idx] = output_image * 255

        buffer_collection.drange = (0, 255)

        # buffer_collection dtype is np.float64 and drange is close but out of [0, 1] before rescaling of any sort
        if self.options.rescaling not in [0, None] and self._is_last_operation:
            buffer_collection = rescale_images255(buffer_collection, rescaling_option=self.options.rescaling)
            # If legacy mode is turned on, rescale_images255 will output uint8
            if not np.issubdtype(buffer_collection.dtype, np.floating):
                for idx, image in enumerate(buffer_collection):
                    buffer_collection[idx] = image.astype('float64')
                buffer_collection.drange = (0, 255)

    def spec_match(self):
        """Match the full Fourier magnitude spectrum to a target spectrum.

        This method replaces the magnitude of each image's Fourier transform with
        ``options.target_spectrum`` while keeping the original phase information.
        It then reconstructs the image through the inverse FFT.

        Notes
        -----
        - This method is used by ``options.mode = 4`` (spectrum-only) and by
          composite histogram+spectrum modes ``6`` and ``8``.
        - The target spectrum must be compatible with the processed image shape
          and channel count.
        - Fourier spectra are computed using the padding parameters from
          ``options.fft_padding_mode`` and ``options.fft_padding_value``.
        - Output values are often outside ``[0, 1]`` in intermediate steps and are
          readjusted when ``self._is_last_operation`` through clipping and
          optional rescaling.
        """
        # Target magnitude spectrum to which the
        # input images should be matched. Should be a 2D or 3D array
        # compatible with the image dimensions, typically of shape
        # (H, W, C).

        if not self._is_first_operation and self.options._is_moving_target and self.options.mode > 4:
            self._compute_initial_spectra()
            self._compute_initial_target_spectrum()
        buffer_collection = self.dataset.buffer

        # If target_spectrum is None, target magnitude is the average of all spectra
        x_size, y_size, n_channels = self._target_spectrum.shape[:3]
        image = im3D(buffer_collection[0])
        if self._target_spectrum.shape[-1] != image.shape[-1]:
            raise TypeError('The target spectrum must have the same number of channels as the images.')

        # Iterate over each image (each entry in the phase collection)
        for idx, image in enumerate(buffer_collection):
            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            console_log(msg=f"\nImage {self._processed_image}", indent_level=0, color=Bcolors.BOLD, verbose=self.verbose>=2)

            matched_image = []
            original_shape = image.shape[:2]

            # Convert the stored phase to 3D array
            phase = im3D(self.dataset.phases[idx])

            # Process each channel separately
            for self._processed_channel in range(n_channels):
                console_log(msg=f"\nChannel {self._processed_channel}", indent_level=1, color=Bcolors.BOLD, verbose=self.verbose>=2)

                # Convert polar (magnitude + phase) back to Cartesian
                XX, YY = pol2cart(self._target_spectrum[:, :, self._processed_channel], phase[:, :, self._processed_channel])

                # Combine into a complex Fourier spectrum
                new = XX + YY * 1j  # 1j = sqrt(-1)

                # Inverse FFT to go back to spatial domain (real-valued image)
                output = np.real(np.fft.ifft2(np.fft.ifftshift(new)))
                matched_image.append(_crop_after_fft(output, original_shape))

                # Comparison: obtained vs target spectrum
                obtained_mag = np.abs(np.fft.fftshift(np.fft.fft2(output)))

                # Flatten for metrics
                t = self._target_spectrum[:, :, self._processed_channel].flatten().astype(np.float64)
                o = obtained_mag.ravel().astype(np.float64, copy=False)

                rmse = compute_rmse(t, o)
                self._validate(observed=[rmse], expected=[0],
                               measures_str=['RMS error'])

            # Soft-clip output values: As this transformation typically produces out-of-range values
            output_image = np.stack(matched_image, axis=-1).squeeze()
            if self._is_last_operation:
                mn, mx = output_image.min(), output_image.max()
                if mn < 0 or mx > 1:
                    console_log(
                        msg=f'Out of range values: Actual range [{mn}, {mx}] outside of the admitted range [0, 1].\nWill be rescaled and clipped so that less than 1% falls outside of [0, 1].',
                        indent_level=1, color=Bcolors.WARNING, verbose=self.verbose==3)
                    output_image = soft_clip(output_image, min_value=0, max_value=1, max_percent=0.01, verbose=self.verbose == 3)

            # Stack the channels and save into the output collection
            buffer_collection[idx] = output_image * 255

        buffer_collection.drange = (0, 255)
        # buffer_collection dtype is np.float64 and drange is close but out of [0, 1] before rescaling of any sort
        if self.options.rescaling not in [0, None] and self._is_last_operation:
            buffer_collection = rescale_images255(buffer_collection, rescaling_option=self.options.rescaling)
            # If legacy mode is turned on, rescale_images255 will output uint8
            if not np.issubdtype(buffer_collection.dtype, np.floating):
                for idx, image in enumerate(buffer_collection):
                    buffer_collection[idx] = image.astype(float)
                buffer_collection.drange = (0, 255)
