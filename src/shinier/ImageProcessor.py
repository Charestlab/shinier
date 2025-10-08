from typing import Optional, List, Union, Iterable, Tuple, Literal
import re
from datetime import datetime
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from shinier import ImageDataset, Options
from shinier.utils import (
    ImageListType, separate, imhist, im3D, cart2pol, pol2cart, soft_clip,
    rescale_images, get_images_spectra, ssim_sens, spectrum_plot, imhist_plot, sf_plot,
    uint8_plus, float01_to_uint, uint_to_float01, noisy_bit_dithering, floyd_steinberg_dithering,
    exact_histogram, exact_histogram_with_noise, Bcolors, MatlabOperators, compute_rmse, RGB2GRAY_WEIGHTS)

Vector = Iterable[Union[float, int]]


class ImageProcessor:
    """Base class for image processing."""

    def __init__(self, dataset: ImageDataset, options: Optional[Options] = None, verbose: Literal[0, 1, 2] = 0):
        self.dataset: ImageDataset = dataset
        self.options: Optional[Options] = options or getattr(dataset, "options", None)
        self.current_masks: Optional[np.ndarray] = None
        self.bool_masks: List = [None] * len(self.dataset.images)
        self.verbose: Literal[-1, 0, 1, 2] = verbose  # -1: Nothing is printed (used for unit tests); 0: Minimal processing steps are printed; 1: Additional info about image and channels being processed are printed; 2: Additional info about the results of internal tests are printed.
        self.log: List = []
        self.validation: List = []
        self.ssim_results: List = []
        self.ssim_data: List = []
        self.seed: int = self.options.seed

        # Private attributes
        self._ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
        self._dataset_map = {
            id(self.dataset.images): 'images',
        }
        if hasattr(self.dataset, 'buffer'):
            self._dataset_map[id(self.dataset.buffer)] = 'buffer'

        self._mode2processing_steps = {
            1: ['lum_match'],
            2: ['hist_match'],
            3: ['sf_match'],
            4: ['spec_match'],
            5: ['hist_match', 'sf_match'],
            6: ['hist_match', 'spec_match'],
            7: ['sf_match', 'hist_match'],
            8: ['spec_match', 'hist_match'],
            9: [None]}
        self._fct_name2process_name = {
            'lum_match': 'luminance matching',
            'hist_match': 'histogram matching',
            'sf_match': 'spatial frequency matching',
            'spec_match': 'fourier spectrum matching',
            None: 'dithering',
        }
        self._iter: int = 0
        self._processing_steps: List[str] = self._mode2processing_steps[self.options.mode]
        self._n_steps: int = len(self._processing_steps)
        self._step: int = 0
        self._processing_function: str
        self._processed_image: str
        self._processed_channel: Optional[int] = None
        self._log_param: dict = {}

        # Run image processing steps
        self.process()
        self.print_log()
        if not self.dataset.images.has_list_array:
            self.dataset.save_images()
            self.dataset.close()
        else:
            self.console_log(
                msg=f'To get the output images, you must instantiate ImageProcessor and call get_results() method. \n\tE.g.: output_images = ImageProcessor(dataset=my_dataset).get_results()',
                level=0, color=Bcolors.WARNING, min_verbose=0)

    def get_results(self):
        """Return list of processed np.ndarray if input was arrays, otherwise None."""
        sp = getattr(self.dataset.images, "src_paths", None)
        if not sp or sp[0] is None:
            return self.dataset.images.data
        return None

    @staticmethod
    def uint8_to_float255(input_collection: ImageDataset, output_collection: ImageDataset) -> ImageDataset:
        """Convert a uintX dataset to float255"""
        for idx, image in enumerate(input_collection):
            output_collection[idx] = image.astype(float)
        output_collection.drange = (0, 255)
        return output_collection

    def _get_mask(self, idx):
        """ Provide mask if masks exists in the dataset, if not make blank masks (all True). """
        n_dims = self.dataset.images.n_dims
        im_size = self.dataset.images.reference_size
        if self.bool_masks[idx] is None:
            if self.options.whole_image == 2:
                self.console_log(msg=f'Preparing mask (whole-image: 2)', level=1, color=Bcolors.HEADER, min_verbose=1)
                self._prepare_mask(image=self.dataset.images[idx])
                self.bool_masks[idx] = np.stack((self.current_masks[0],) * (3 if n_dims == 3 else 1), axis=-1).squeeze()
            elif self.options.whole_image == 3:
                if idx < self.dataset.n_masks:  # If there is one mask, it picks self.bool_masks[0] everytime
                    self.console_log(msg=f'Preparing mask (whole-image: 3)', level=1, color=Bcolors.HEADER,
                                     min_verbose=1)
                    self._prepare_mask(image=self.dataset.images[idx], mask=self.dataset.masks[idx])
                    self.bool_masks[idx] = np.stack((self.current_masks[0],) * (3 if n_dims == 3 else 1),
                                                    axis=-1).squeeze()
                else:
                    self.bool_masks[idx] = self.bool_masks[0]
            else:
                masks_size = im_size
                if n_dims == 3:
                    masks_size = list(im_size) + [n_dims]
                self.bool_masks[idx] = np.ones(masks_size, dtype=bool).squeeze() if idx == 0 else self.bool_masks[0]

    def _prepare_mask(self, image, mask=None):
        if self.options.whole_image == 2:
            mask_f, mask_b, _ = separate(image, self.options.background)
        elif self.options.whole_image == 3:
            mask_f, mask_b, _ = separate(mask, self.options.background)
        self.current_masks = (mask_f, mask_b)

    def _validate_ssim(self, ssim: List[float]):
        if len(ssim) > 1:
            out = np.mean(ssim, axis=1)
            is_strictly_increasing = np.all(np.diff(out) > -1e-3)
            res_color = Bcolors.OKGREEN if is_strictly_increasing else Bcolors.FAIL
            res_txt = 'PASS' if is_strictly_increasing else 'FAIL'
            res = f'{Bcolors.OKCYAN}SSIM optimization test:{Bcolors.ENDC} {res_color}{res_txt}{Bcolors.ENDC}'
            self.console_log(msg=res, level=1, min_verbose=1)
            results = {
                'iter': self._iter,
                'step': self._step,
                'image': self._processed_image,
                'valid_result': is_strictly_increasing
            }
            self.ssim_results.append(results)

    def _validate(self, observed: List[float], expected: List[float], measures_str: list[str], tolerance: float = 5e-3):
        """Internal validation"""
        if len(observed) != len(expected) or len(observed) != len(measures_str):
            raise ValueError('observed, expected and measures_str lists must be the same size')
        diff = [np.abs(obs - expected[idx]) for idx, obs in enumerate(observed)]
        results = {
            'iter': self._iter,
            'step': self._step,
            'processing_function': self._processing_function,
            'image': self._processed_image,
            'channel': self._processed_channel,
            'valid_result': np.all([d < tolerance for d in diff])
        }
        obs = ', '.join([f'{msr} = {observed[idx]:4.4f}' for idx, msr in enumerate(measures_str)])
        obs = f'Observed: {obs}'
        exp = ', '.join([f'{msr} = {expected[idx]:4.4f}' for idx, msr in enumerate(measures_str)])
        exp = f'Expected: {exp}'
        res_color = Bcolors.OKGREEN if results['valid_result'] else Bcolors.FAIL
        res_txt = 'PASS' if results['valid_result'] else 'FAIL'
        res = f'{Bcolors.OKCYAN}Internal test:{Bcolors.ENDC} {res_color}{res_txt}{Bcolors.ENDC}'
        if res_txt == 'FAIL' and self.verbose > 1:
            print(res)
            raise Exception(f"At least one difference between expected and observed values is larger than tolerance: {diff}")
        results['log_result'] = f'{Bcolors.OKBLUE}{obs}\n{exp}{Bcolors.ENDC}\n{res}'
        indent_level = 1 if self._processed_channel is None else 2
        self.console_log(msg=results['log_result'], level=indent_level, min_verbose=2)
        self.validation.append(results)

    def console_log(self, msg: str, level: int = 0, color: Optional[str] = None, min_verbose: int = 1):
        def _set_indent_and_color(text, lev: int, col: Optional[str] = None):
            indent_str = '\t' * lev
            if col is not None:
                return "\n".join(f'{indent_str}{col}{line}{Bcolors.ENDC}' for line in text.splitlines())
            else:
                return "\n".join(f'{indent_str}{line}' for line in text.splitlines())

        # Log message
        msg = _set_indent_and_color(msg, level, color)
        self.log.append(msg)
        if self.verbose >= min_verbose:
            print(msg)

    def print_log(self) -> None:
        """ Record processing_steps list for reproducibility """
        # Generate a filename with the full date and time
        def _strip_ansi(s: str) -> str:
            return self._ANSI_RE.sub("", s)

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = Path(self.options.output_folder) / f"log_{current_datetime}.txt"

        # Write each step to a new line in the file
        with open(filename, 'w') as file:
            for step in self.log:
                file.write(_strip_ansi(step) + '\n')

    def process(self):
        """
        Applies all steps of image processing pertaining to a given mode.
        All processing functions assume float images with a [0, 255] range.
        """

        # Put input images into buffer dataset and convert to float [0, 255]
        self.dataset.buffer = self.uint8_to_float255(self.dataset.images, self.dataset.buffer)

        if self.options.mode in [2, 5, 6, 7, 8] and self.options.hist_specification:
            # Set a seed for the random generator used in exact histogram specification
            if self.seed is None:
                now = datetime.now()
                self.seed = int(now.timestamp())
            np.random.seed(self.seed)
            self.log.append(f'seed={self.seed}')
            self.console_log(msg=f'Use this seed for reproducibility: {self.seed}', color=Bcolors.WARNING, level=0, min_verbose=0)

        # A first loop runs n times the processing steps associated with given mode.
        # A second loop is for modes associated with multiple steps will run more
        for self._iter in range(self.options.iterations):
            for self._step, self._processing_function in enumerate(self._processing_steps):
                if self._processing_function is not None:
                    # Get the processing function, check and call it
                    exec_fct = getattr(self, self._processing_function, None)
                    if exec_fct is None:
                        raise RuntimeError(f'Function {self._processing_function} does not exist in ImageProcessor class')

                    self.console_log(
                        msg=f'Applying {self._fct_name2process_name[self._processing_function]}... (iter={self._iter}, step={self._step})',
                        level=0,
                        color=Bcolors.SECTION,
                        min_verbose=0
                    )
                    exec_fct()
                    print('') if self.verbose > 0 else None

        # Applies dithering or simply convert into uint8 if no dithering
        self.dataset.images = self.dithering(
            input_collection=self.dataset.buffer,
            output_collection=self.dataset.images,
            dithering=self.options.dithering)

    def dithering(self, input_collection: ImageListType, output_collection: ImageListType, dithering: Literal[0, 1, 2]):
        # Dithering function assumes float with values in the [0, 1] range.
        for idx, image in enumerate(input_collection):
            if dithering == 1:  # Make sure images are float01
                output_collection[idx] = noisy_bit_dithering(image=image/255, depth=256, legacy_mode=self.options.legacy_mode)
            elif dithering == 2:
                output_collection[idx] = floyd_steinberg_dithering(image=image/255, depth=256, legacy_mode=self.options.legacy_mode)
            else:
                output_collection[idx] = MatlabOperators.uint8(image) if self.options.legacy_mode else uint8_plus(image=image, verbose=self.verbose>0)

        return output_collection

    def lum_match(self):
        """
        Matches the mean and standard deviation of a set of images. If target_lum is provided, it will match the mean and standard
        deviation of target_lum, where target_lum[0] is the mean and target_lum[1] is the standard deviation. If safe_values is enabled, it will
        find a target mean and standard deviation that is close to target_lum while not producing out-of-range values, i.e. outside of [0, 255].

            Warnings:
                - Clipping should be applied prior to uint8 conversion since np.uint8 and .astype('uint8') exhibit wrap-around behavior for out-of-range values. E.g. np.array([-2, 256]).astype('uint8') = [254, 0]
                - the target M and STD provided if safe_values is true, will not be equal to the grand average of the images' mean and std. Instead, it will find the closet mean and std that prevent out-of-range values.
        """

        def predict_values(original_means, original_stds, original_min_max, target_mean, target_std):
            predicted_min = (np.array(original_min_max)[:, 0] - np.array(original_means)) / np.array(original_stds) * target_std + target_mean
            predicted_max = (np.array(original_min_max)[:, 1] - np.array(original_means)) / np.array(original_stds) * target_std + target_mean
            predicted_range = predicted_max - predicted_min
            return predicted_min, predicted_max, predicted_range

        def compute_m_and_sd(image: np.ndarray, binary_mask: np.ndarray) -> Tuple[float, float]:
            """
            M and SD is a weighted sum of the channels for RGB images. Normal otherwise.
            Args:
                image: An image
                binary_mask: A mask

            Returns:
                Tuple: mean, standard deviation
            """

            if self.options.as_gray != 0:
                M = MatlabOperators.mean2(im[binary_mask]) if self.options.legacy_mode else np.mean(im[binary_mask])
                SD = MatlabOperators.mean2(im[binary_mask]) if self.options.legacy_mode else np.mean(im[binary_mask])
            else:
                convertion_type = RGB2GRAY_WEIGHTS['int2key'][self.options.rgb_weights]
                ch_weights = RGB2GRAY_WEIGHTS[convertion_type]
                ch_means = np.array([np.mean(im[:, :, c][binary_mask[:, :, c]]) for c in range(3)])
                ch_stds = np.array([np.std(im[:, :, c][binary_mask[:, :, c]]) for c in range(3)])
                M = np.sum(ch_means * ch_weights)
                SD = np.sqrt(np.sum((ch_weights ** 2) * (ch_stds ** 2)))

            return M, SD

        buffer_collection = self.dataset.buffer

        # 1) Compute the mean and standard deviation of the original images.
        # 2) Compute the target mean and standard deviation if not provided.
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
            self._get_mask(idx)
            M, SD = compute_m_and_sd(image=im, binary_mask=self.bool_masks[idx])
            original_means.append(M)
            original_stds.append(SD)
            original_min_max.append((im[self.bool_masks[idx]].min(), im[self.bool_masks[idx]].max()))
        target_mean, target_std = (np.mean(original_means), np.mean(original_stds)) if target_lum == (0, 0) else target_lum
        predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)

        if safe_values and (any(predicted_min < 0) or any(predicted_max > 255)):
            max_range = predicted_max.max() - predicted_min.min()
            scaling_factor = min(1, (255 - 1e-6) / max_range)  # Safety margin of 1e-6 to avoid precision issues
            target_std *= scaling_factor
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            target_mean = target_mean + (255 - np.max(predicted_max))
            self.console_log(msg=f"Adjusted target values for safe values: M = {target_mean:.4f}, SD = {target_std:.4f}", level=0,color=Bcolors.WARNING, min_verbose=0)
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            if np.any(predicted_min < -1e-3) or np.any(predicted_max > (255 + 1e-3)):
                raise Exception(f'Out-of-range values detected: mins = {list(predicted_min)}, maxs = {list(predicted_max)}')

        for idx, im in enumerate(buffer_collection):
            im2 = im.copy()
            M, SD = compute_m_and_sd(image=im2, binary_mask=self.bool_masks[idx])

            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            self.console_log(msg=f"Image {self._processed_image}:", level=0, color=Bcolors.BOLD, min_verbose=1)
            self.console_log(msg=f"Original: M = {M:.4f}, SD = {SD:.4f}", level=1, color=Bcolors.OKBLUE, min_verbose=2)

            # Standardization
            if original_stds[idx] != 0:
                im2[self.bool_masks[idx]] = (im2[self.bool_masks[idx]] - original_means[idx]) / original_stds[idx] * target_std + target_mean
            else:
                im2[self.bool_masks[idx]] = target_mean

            M, SD = compute_m_and_sd(image=im2, binary_mask=self.bool_masks[idx])

            # Save resulting image
            self.console_log(msg=f"Target values: M = {target_mean:.4f}, SD = {target_std:.4f}", level=1, color=Bcolors.OKBLUE, min_verbose=2)
            self.dataset.buffer[idx] = im2  # update the dataset
            self._validate(observed=[M, SD], expected=[target_mean, target_std], measures_str=['M', 'SD'])

        self.dataset.buffer.drange = (0, 255)

    def hist_match(self):
        """
        Equates a set of images in terms of luminance histograms.
        """

        def _avg_hist(images: ImageListType, normalized: bool = True, n_bins: int = 256) -> np.ndarray:
            """Computes the average histogram of a set of images.

            Args:
                images (ImageListType): A list of images
                normalized (bool): Indicate of the result should be normalize to sum to 1.
                n_bins (int): Number of levels in the image (uint8 = 256)

            Returns:
                average (np.ndarray): Average histogram counts for each channel.

            Notes:
                This function cannot be externalize due to the use of _get_mask().
            """
            n_channels = 1 if images.n_dims == 2 else 3
            # n_bins = max(images.drange) + 1 if not np.issubdtype(images.dtype, np.floating) else 256  # TODO: Imposing 256 but is this ok for all use case?
            hist_sum = np.zeros((n_bins, n_channels))
            for idx, im in enumerate(images):
                self._get_mask(idx)  # Prepare the mask if it does not already exist. Default mask is all True.
                hist_sum += imhist(im, self.bool_masks[idx], n_bins=n_bins)

            # Average of the pixels in the bins
            average = hist_sum / len(images)
            if normalized:
                average = average / (average.sum(axis=0, keepdims=True) + 1e-12)
                # average /= average.sum()

            return average

        noise_level = 0.1  # Noise level to be applied in case hist_specification is set to 1. Default is 0.1.
        target_hist = self.options.target_hist
        hist_optim = self.options.hist_optim
        hist_specification = self.options.hist_specification

        # Get appropriate image collection
        input_collection = self.dataset.buffer.readonly_copy()
        buffer_collection = self.dataset.buffer

        if target_hist is None and (self.options.mode in [5, 6, 7, 8] or hist_optim == 1):
            bit_size = 16
        elif target_hist is not None:
            n_bins = target_hist.shape[0]
            if n_bins not in [256, 65536]:
                raise ValueError('Target hist must contain either 256 or 65536 elements')
            bit_size = int(np.log2(n_bins))
        else:
            bit_size = 8
        n_bins = 2 ** bit_size

        if buffer_collection.drange[1] < n_bins:
            for idx, image in enumerate(input_collection):
                buffer_collection[idx] = image / buffer_collection.drange[1] * n_bins
        buffer_collection.drange = (0, n_bins)

        if target_hist is None:
            target_hist = _avg_hist(buffer_collection, n_bins=n_bins)  # Placeholder for avgHist
        else:
            for idx, im in enumerate(buffer_collection):
                self._get_mask(idx)
            if target_hist.shape[0] != n_bins:
                raise ValueError(f"target_hist must have {n_bins} bins, but has {target_hist.shape[0]}.")
            if target_hist.max()>1:
                target_hist = target_hist.astype(np.float64)
                target_hist /= (target_hist.sum(axis=0, keepdims=True) + 1e-12)

        # If hist_optim disable, will run only one loop (n_iter = 1)
        n_iter = self.options.hist_iterations + 1 if hist_optim else 1  # See important note below to explain the +1. Also, note that the number of iterations for SSIM optimization (default = 10)
        step_size = self.options.step_size  # Step size (default = 34)

        # Match the histogram
        self._processed_channel = None
        for idx, image in enumerate(buffer_collection):
            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            self.console_log(msg=f"Image {self._processed_image}:", level=0, color=Bcolors.BOLD, min_verbose=1)

            image = im3D(image)
            X = image
            M = np.prod(image.shape[:2])
            all_ssim = []
            for self._sub_iter in range(n_iter):  # n_iter = 1 when hist_optim == 0
                if n_iter > 1 and self._sub_iter < n_iter - 1:
                    self.console_log(msg=f"Optimization (iter={self._sub_iter + 1}):", level=1, color=Bcolors.BOLD, min_verbose=1)

                if hist_specification == 1:
                    Y = exact_histogram_with_noise(image=X, binary_mask=self.bool_masks[idx], target_hist=target_hist, noise_level=noise_level, n_bins=n_bins)
                else:
                    Y, OA = exact_histogram(image=X, target_hist=target_hist, binary_mask=self.bool_masks[idx], n_bins=n_bins, verbose=self.verbose>=1)
                    if n_iter == 1 or (n_iter > 1 and self._sub_iter < n_iter - 1):
                        self.console_log(msg=f"Ordering accuracy per channel = {OA}", level=1, color=Bcolors.OKBLUE, min_verbose=2)
                # sens, ssim = ssim_sens(image, Y, n_bins=n_bins)
                sens, ssim = ssim_sens(image/n_bins, Y/n_bins, n_bins=2)
                if n_iter > 1 and self._sub_iter < n_iter - 1:
                    all_ssim.append(ssim)

                if hist_optim and (n_iter == 1 or (n_iter > 1 and self._sub_iter < n_iter - 1)):
                    self.console_log(msg=f"Mean SSIM = {np.mean(ssim):.4f}", level=1, color=Bcolors.OKBLUE, min_verbose=2)
                if hist_optim and self._sub_iter < n_iter - 1:
                    ssim_update = sens * step_size * M
                    X = Y + ssim_update  # X float64, Y uint8/uint16
                    X = np.rint(np.clip(X, 0, 2 ** bit_size - 1)).astype(f'uint{bit_size}')

            # Test monotonic increase of ssim between first and last iteration
            if self.options.hist_optim and len(all_ssim) >=2:
                self._validate_ssim(ssim=[all_ssim[0], all_ssim[-1]])

            # Important Note:
            # - Must use Y as this is the one that matches the target histogram.
            # - n_iter is adjusted to +1 to assure the proper number of optimization steps is run.
            # new_image = np.rint(np.clip(Y, 0, 2 ** bit_size - 1)).astype(f'uint{bit_size}')
            new_image = Y.copy()

            # Make sure the output is always float255
            buffer_collection[idx] = Y / n_bins * 255

            # Compute statistics
            final_hist = imhist(image=new_image, mask=self.bool_masks[idx], n_bins=n_bins, normalized=True)
            # plt.plot(np.stack([final_hist[:, 0], target_hist[:, 0]]).T)
            corr = np.corrcoef(final_hist.flatten(), target_hist.flatten())
            rmse = compute_rmse(final_hist.flatten(), target_hist.flatten())
            self.console_log(msg=f"SSIM index between transformed and original image: {np.mean(ssim):.4f}", level=1, color=Bcolors.OKBLUE, min_verbose=2)
            self._validate(observed=[corr[0, 1], rmse], expected=[1, 0], measures_str=['correlation (target vs obtained histogram count)', 'RMS error (target vs obtained histogram count)'])

        buffer_collection.drange = (0, 255)

    def sf_match(self):
        """Match spatial frequencies of input images to a target rotational spectrum.

        This function performs spatial frequency (SF) matching by adjusting the
        rotational average of the Fourier amplitude of each input image so that
        it matches the target spectrum. Each input image's magnitude spectrum
        is scaled relative to the target spectrum, while preserving its original
        phase, and then reconstructed in the spatial domain.

        Notes:
            - get_images_spectra will stretch input to [0, 1] range
            - Frequencies beyond the Nyquist limit are set to zero to avoid aliasing.
            - The adjustment is performed separately for each channel.
            - Uses `cart2pol` and `pol2cart` to switch between Cartesian and polar
              representations of the Fourier domain.

        """

        def rot_avg(arr2d: np.ndarray, radius: np.ndarray) -> np.ndarray:
            """Mean magnitude per radius bin (annular average)."""
            sums = np.bincount(radius, weights=arr2d.ravel())
            return sums / ann_counts

        # Target magnitude spectrum to which the
        # input images should be matched. Should be a 2D or 3D array
        # compatible with the image dimensions, typically of shape
        # (H, W, C).
        target_spectrum = self.options.target_spectrum

        # Get proper input and output image collections
        # input_collection, input_name, output_collection, output_name = self._get_relevant_input_output()
        buffer_collection = self.dataset.buffer
        for idx, image in enumerate(buffer_collection):
            buffer_collection[idx] = image/255
        buffer_collection.drange = (0, 1)

        # Compute all spectra: Assumes float01
        self.dataset.magnitudes, self.dataset.phases = get_images_spectra(
            images=buffer_collection,
            magnitudes=self.dataset.magnitudes,
            phases=self.dataset.phases, rescale=self.options.rescaling)

        # If target_spectrum is None, target magnitude is the average of all spectra
        if target_spectrum is None:
            target_spectrum = np.zeros(self.dataset.magnitudes[0].shape)
            for idx, mag in enumerate(self.dataset.magnitudes):
                target_spectrum += mag
            target_spectrum /= len(self.dataset.magnitudes)
        else:
            if target_spectrum.shape[:2] != self.dataset.images.reference_size:
                raise TypeError('The target spectrum must have the same size as the images.')

        target_spectrum = im3D(target_spectrum)
        x_size, y_size, n_channels = target_spectrum.shape[:3]

        #  Returns the frequencies of the image, bins range from -0.5f to 0.5f (0.5f is the Nyquist frequency) 1/y_size is the distance between each pixel in the image
        # f1 = np.fft.fftshift(np.fft.fftfreq(x_size, d=1 / x_size))
        # f2 = np.fft.fftshift(np.fft.fftfreq(y_size, d=1 / y_size))
        f_cols = np.fft.fftshift(np.fft.fftfreq(y_size, d=1 / y_size))  # like f1 in MATLAB
        f_rows = np.fft.fftshift(np.fft.fftfreq(x_size, d=1 / x_size))  # like f2 in MATLAB
        XX, YY = np.meshgrid(f_cols, f_rows)
        nyquistLimit = np.floor(max(x_size, y_size) / 2)
        # XX, YY = np.meshgrid(f1, f2)
        r, theta = cart2pol(XX, YY)

        # Map of the bins of the frequencies
        r = MatlabOperators.round(r) if self.options.legacy_mode else np.round(r, decimals=0)

        # Need to be a 1D array of integers for the bincount function
        r_int = r.astype(np.int32)
        r1 = r_int.ravel()

        # Precompute counts per radius (for true rotational *averages*, not sums)
        ann_counts = np.bincount(r1)
        ann_counts[ann_counts == 0] = 1  # protect against divide-by-zero

        # Match spatial frequency on rotational average of the magnitude spectrum
        for idx, image in enumerate(buffer_collection):
            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            self.console_log(msg=f"Image {self._processed_image}:", level=0, color=Bcolors.BOLD, min_verbose=1)
            matched_image = []
            magnitude = im3D(self.dataset.magnitudes[idx])
            phase = im3D(self.dataset.phases[idx])
            for self._processed_channel in range(n_channels):
                self.console_log(msg=f'Channel {self._processed_channel}:', level=1, min_verbose=2, color=Bcolors.BOLD)
                fft_image = magnitude[:, :, self._processed_channel]

                # Rotational averages (target vs source) as MEANS over annuli
                target_ra = rot_avg(target_spectrum[:, :, self._processed_channel], radius=r1)
                source_ra = rot_avg(fft_image, radius=r1)

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

                # # Keep original DC: Should minimize out-of-range values.
                # mu = float(np.mean(image[:, :, self._processed_channel]))
                # H, W = new.shape
                # cy, cx = H // 2, W // 2  # DC bin in *shifted* spectrum
                # desired_DC = mu * (H * W)  # complex DC (real, since it's a mean)
                # new[cy, cx] = desired_DC + 0.0j

                output = np.real(np.fft.ifft2(np.fft.ifftshift(new)))
                matched_image.append(output)

                # Comparison: obtained vs target rotational averages (up to Nyquist)
                obtained_ra = rot_avg(new_magnitude, radius=r1)
                R = int(min(len(obtained_ra), len(target_ra), nyquistLimit + 1))
                t = target_ra[:R]
                o = obtained_ra[:R]
                corr = np.corrcoef(t, o)[0, 1] if R > 1 else np.nan
                rmse = compute_rmse(t, o)
                self._validate(observed=[corr, rmse], expected=[1, 0],
                               measures_str=['correlation (target vs obtained rotational average)',
                                             'RMS error (target vs obtained rotational average)'])

            buffer_collection[idx] = np.stack(matched_image, axis=-1).squeeze() * 255

        buffer_collection.drange = (0, 255)
        # buffer_collection dtype is np.float64 and drange is close but out of [0, 1] before rescaling of any sort
        # TODO: NEEDS TO BE CHECKED
        if self.options.rescaling:
            buffer_collection = rescale_images(buffer_collection, rescaling_option=self.options.rescaling)
            # If legacy mode is turned on, rescale_images will output uint8
            if not np.issubdtype(buffer_collection.dtype, np.floating):
                for idx, image in enumerate(buffer_collection):
                    buffer_collection[idx] = image.astype(float)
                buffer_collection.drange = (0, 255)

    def spec_match(self):
        """Match the full magnitude spectrum of images to a target spectrum.

        This function reconstructs images whose Fourier magnitude is replaced
        by the `target_spectrum`, while preserving the original Fourier phase.
        The inverse FFT is then used to obtain spatial-domain images with the
        desired spectral characteristics.

        Notes:
            - Phase information from each input image is preserved.
            - The output is real-valued because only magnitude is replaced.

        """
        # Target magnitude spectrum to which the
        # input images should be matched. Should be a 2D or 3D array
        # compatible with the image dimensions, typically of shape
        # (H, W, C).
        target_spectrum = self.options.target_spectrum

        # Get proper input and output image collections
        buffer_collection = self.dataset.buffer
        for idx, image in enumerate(buffer_collection):
            buffer_collection[idx] = image/255
        buffer_collection.drange = (0, 1)

        # Compute all spectra
        self.dataset.magnitudes, self.dataset.phases = get_images_spectra(
            images=buffer_collection,
            magnitudes=self.dataset.magnitudes,
            phases=self.dataset.phases)

        # If target_spectrum is None, target magnitude is the average of all spectra
        if target_spectrum is None:
            target_spectrum = np.zeros(self.dataset.magnitudes[0].shape)
            for idx, mag in enumerate(self.dataset.magnitudes):
                target_spectrum += mag
            target_spectrum /= len(self.dataset.magnitudes)
        else:
            if target_spectrum.shape != self.dataset.images.reference_size:
                raise TypeError('The target spectrum must have the same size as the images.')

        # Ensure the target spectrum is 3D (H, W, C)
        target_spectrum = im3D(target_spectrum)
        x_size, y_size, n_channels = target_spectrum.shape[:3]

        # Iterate over each image (each entry in the phase collection)
        for idx, image in enumerate(buffer_collection):
            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            self.console_log(msg=f"Image {self._processed_image}:", level=0, color=Bcolors.BOLD, min_verbose=1)

            matched_image = []

            # Convert the stored phase to 3D array
            phase = im3D(self.dataset.phases[idx])

            # Process each channel separately
            for self._processed_channel in range(n_channels):
                self.console_log(msg=f"Channel {self._processed_channel}:", level=1, color=Bcolors.BOLD, min_verbose=2)

                # Convert polar (magnitude + phase) back to Cartesian
                XX, YY = pol2cart(target_spectrum[:, :, self._processed_channel], phase[:, :, self._processed_channel])

                # Combine into a complex Fourier spectrum
                new = XX + YY * 1j  # 1j = sqrt(-1)

                # Inverse FFT to go back to spatial domain (real-valued image)
                output = np.real(np.fft.ifft2(np.fft.ifftshift(new)))

                matched_image.append(output)

                # Comparison: obtained vs target spectrum
                obtained_mag = np.abs(np.fft.fftshift(np.fft.fft2(output)))

                # Flatten for metrics
                t = target_spectrum[:, :, self._processed_channel].flatten().astype(np.float64)
                o = obtained_mag.ravel().astype(np.float64, copy=False)

                corr = float(np.corrcoef(t, o)[0, 1]) if t.size > 1 else float('nan')
                rmse = compute_rmse(t, o)
                self._validate(observed=[corr, rmse], expected=[1, 0],
                               measures_str=['correlation (target vs obtained spectrum)',
                                             'RMS error (target vs obtained spectrum)'])

            # Stack the channels and save into the output collection
            buffer_collection[idx] = np.stack(matched_image, axis=-1).squeeze() * 255

        buffer_collection.drange = (0, 255)
        # buffer_collection dtype is np.float64 and drange is close but out of [0, 1] before rescaling of any sort
        # TODO: NEEDS TO BE CHECKED
        if self.options.rescaling:
            buffer_collection = rescale_images(buffer_collection, rescaling_option=self.options.rescaling)
            # If legacy mode is turned on, rescale_images will output uint8
            if not np.issubdtype(buffer_collection.dtype, np.floating):
                for idx, image in enumerate(buffer_collection):
                    buffer_collection[idx] = image.astype(float)
                buffer_collection.drange = (0, 255)
