from typing import Optional, List, Union, Iterable, Tuple, Literal
from datetime import datetime
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from shinier import ImageDataset, Options
from shinier.utils import (
    beta_bounds_from_ssim, ImageListType, separate, imhist, im3D, cart2pol, pol2cart, soft_clip,
    rescale_images255, get_images_spectra, ssim_sens, spectrum_plot, imhist_plot, sf_plot, avg_hist,
    uint8_plus, float01_to_uint, uint_to_float01, noisy_bit_dithering, floyd_steinberg_dithering,
    exact_histogram, Bcolors, MatlabOperators, compute_rmse, RGB2GRAY_WEIGHTS,
    has_duplicates, stretch, console_log, print_log
)
import sys

Vector = Iterable[Union[float, int]]


class ImageProcessor:
    """
    Provides functionality for image processing with multiple configurable steps.

    This class is designed to process a dataset of images by applying the following
    configurable transformations: luminance matching, histogram matching, spatial frequency
    matching, and Fourier spectrum matching. It is highly customizable and includes
    options like verbosity levels, random seed initialization, and mask generation
    for more fine-grained processing. Validation test are applied on each transformation
    and all processing steps and important information (e.g. seed) are logged in the output_folder.

    Attributes:
        dataset (ImageDataset): The dataset containing the images to be processed.
        options (Optional[Options]): Options for processing, including mode, verbosity,
            and seed values.
        current_masks (Optional[np.ndarray]): Masks for the current image being processed.
        bool_masks (List): Boolean masks for all images in the dataset.
        verbose (Literal[-1, 0, 1, 2, 3]): Controls verbosity levels (default = 0).
            -1 = Quiet mode
            0 = Progress bar with ETA
            1 = Basic progress steps (no progress bar)
            2 = Additional info about image and channels being processed are printed (no progress bar)
            3 = Debug mode for developers (no progress bar)
        log (List): Log messages related to processing.
        validation (List): Results of validation checks during processing.
        ssim_results (List): Structural Similarity Index (SSIM) test results.
        ssim_data (List): SSIM-related data for analysis.
        seed (int): Random seed for reproducibility in processing steps.

        Notes:
            - Using exact_histogram_without_ties whenever possible for faster and deterministic results.
            - Input images are transformed into floats [0, 255] at the very beginning. A buffer image dataset
                is used to store intermediate results. Output images are reconverted into uint8.
            - All important processing information (e.g. seed) are logged and stored in output_folder.
    """

    def __init__(self, dataset: ImageDataset, options: Optional[Options] = None, verbose: Optional[Literal[-1, 0, 1, 2, 3]] = None, from_cli: bool = False):
        self.dataset: ImageDataset = dataset
        self.options: Optional[Options] = options or getattr(dataset, "options", None)
        self.current_masks: Optional[np.ndarray] = None
        self.bool_masks: List = [None] * len(self.dataset.images)
        self.verbose: Literal[-1, 0, 1, 2, 3] = verbose if verbose is not None else self.options.verbose  # -1: Nothing is printed (used for unit tests); 0: Minimal processing steps are printed; 1: Additional info about image and channels being processed are printed; 2: Additional info about the results of internal tests are printed.
        self.log: List = []
        self.validation: List = []
        self.ssim_results: List = []
        self.ssim_data: List = []
        self.seed: int = self.options.seed

        # Private attributes
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
        self._is_last_operation: bool = False
        self._sum_bool_masks: List = [None] * len(self.dataset.images)
        self._from_cli: bool = from_cli
        # Run image processing steps
        self.process()
        print_log(logs=self.log, log_path=self.options.output_folder)
        if not self.dataset.images.has_list_array:
            self.dataset.save_images()
            self.dataset.close()
        else:
            console_log(
                msg=f'To get the output images, you must instantiate ImageProcessor and call get_results() method. \n\tE.g.: output_images = ImageProcessor(dataset=my_dataset).get_results()',
                indent_level=0, color=Bcolors.WARNING, verbose=self.verbose >= 2 and not self._from_cli)

    def get_results(self):
        """Return list of processed np.ndarray if input was arrays, otherwise None."""
        sp = getattr(self.dataset.images, "store_paths", None)
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
        background = 127 if self.options.whole_image == 3 and self.options.background == 300 else self.options.background
        background_operator = '<' if self.options.whole_image == 3 and self.options.background == 300 else '=='
        def _prepare_mask(image, mask=None):
            if self.options.whole_image == 2:
                mask_f, mask_b, _ = separate(image, background=background, background_operator=background_operator)
            elif self.options.whole_image == 3:
                mask_f, mask_b, _ = separate(mask, background=background, background_operator=background_operator)
            self.current_masks = (mask_f, mask_b)

        n_dims = self.dataset.images.n_dims
        im_size = np.array(list(self.dataset.images.reference_size) + [self.dataset.images.n_channels])
        if self.bool_masks[idx] is None:
            if self.options.whole_image == 2:
                console_log(msg=f'Preparing mask (whole-image: 2)', indent_level=0, color=Bcolors.HEADER, verbose=self.verbose > 1)
                _prepare_mask(image=self.dataset.images[idx])
                self.bool_masks[idx] = np.stack((self.current_masks[0],) * (3 if n_dims == 3 else 1), axis=-1)
            elif self.options.whole_image == 3:
                if idx < self.dataset.n_masks:  # If there is one mask, it picks self.bool_masks[0] everytime
                    console_log(msg=f'Preparing mask (whole-image: 3)', indent_level=0, color=Bcolors.HEADER, verbose=self.verbose > 1)
                    _prepare_mask(image=self.dataset.images[idx], mask=self.dataset.masks[idx])
                    self.bool_masks[idx] = np.stack((self.current_masks[0],) * (3 if n_dims == 3 else 1), axis=-1)
                else:
                    self.bool_masks[idx] = self.bool_masks[0]
            else:
                # No ROI mask: Whole image is analyzed
                self.bool_masks[idx] = np.ones(im_size, dtype=bool) if idx == 0 else self.bool_masks[0]
            self._sum_bool_masks[idx] = [self.bool_masks[idx][..., ch].sum() for ch in range(self.bool_masks[idx].shape[2])]

    def _validate_ssim(self, ssim: List[float]):
        out = np.array(ssim)
        if out.shape[0] > 1:
            for ch in range(out.shape[1]):
                is_strictly_increasing = np.all(np.diff(out[:, ch], axis=0) > -1e-3)
                results = {
                    'iter': self._iter,
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
            'iter': self._iter,
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
        """
        The method consists of a series of image processing operations which are applied to the dataset's
        images buffer. The input uint8 images are first converted into float [0, 255]. Dithering and
        reconversion into uint8 is applied in the final step, based on the provided dithering option.

        If certain processing modes support histogram specification, a reproducible random seed is optionally generated
        and logged for replicability.
        """

        # Put input images into buffer dataset and convert to float [0, 255]
        self.dataset.buffer = self.uint8_to_float255(self.dataset.images, self.dataset.buffer)

        # Set a seed for the random generator used in exact histogram specification
        if self.seed is None:
            now = datetime.now()
            self.seed = int(now.timestamp())
        np.random.seed(self.seed)
        self.log.append(f'seed={self.seed}')
        console_log(msg=f'Use this seed for reproducibility: {self.seed}', color=Bcolors.WARNING, indent_level=0, verbose=self.verbose>=1 and not self._from_cli)

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

        # A first loop runs n times the processing steps associated with given mode.
        # A second loop is for modes associated with multiple steps will run more
        mask_prepared = False
        cnt = 0
        for self._iter in range(self.options.iterations):
            for self._step, self._processing_function in enumerate(self._processing_steps):
                self._is_last_operation = self._iter == self.options.iterations - 1 and self._step == self._n_steps - 1
                if self._processing_function is not None:
                    # Prepare masks if not done
                    if not mask_prepared:
                        for idx in range(self.dataset.images.n_images):
                            self._get_mask(idx)
                            mask_prepared = True

                    # Get the processing function, check and call it
                    if self.verbose == 0:
                        pbar.set_description(f'{Bcolors.ALMOST_WHITE}Applying {self._fct_name2process_name[self._processing_function]}... (iter={self._iter}, step={self._step}){Bcolors.ENDC}')

                    exec_fct = getattr(self, self._processing_function, None)
                    if exec_fct is None:
                        raise RuntimeError(f'Function {self._processing_function} does not exist in ImageProcessor class')

                    console_log(
                        msg=f'Applying {self._fct_name2process_name[self._processing_function]}... (iter={self._iter}, step={self._step})',
                        indent_level=0,
                        color=Bcolors.SECTION,
                        verbose=self.verbose>=1
                    )
                    exec_fct()
                    print('') if self.verbose > 1 else None  # Adds \n between process
                    if self.verbose == 0:
                        pbar.update(1)
                        cnt+=1

        # Applies dithering or simply convert into uint8 if no dithering
        self.dataset.images = self.dithering(
            input_collection=self.dataset.buffer,
            output_collection=self.dataset.images,
            dithering=self.options.dithering)

    def dithering(self, input_collection: ImageListType, output_collection: ImageListType, dithering: Literal[0, 1, 2]):
        """
        Applies a dithering effect to a collection of images based on the specified dithering mode.

        This function processes each image in the input collection individually and applies the appropriate
        dithering method. Supported methods include noisy bit dithering, Floyd-Steinberg dithering, and
        a standard uint8 conversion, depending on the specified mode. The processed images are stored
        in the provided output collection.

        Args:
            input_collection: A collection of input images to be processed.
            output_collection: A collection where the processed images will be stored.
            dithering: An integer indicating the dithering method to use. 1 corresponds
                to noisy bit dithering, 2 corresponds to Floyd-Steinberg dithering, and
                any other value defaults to a standard uint8 conversion.

        Returns:
            A collection with the results of the applied dithering effect.
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
        """
        Matches the mean and standard deviation of a set of images. If target_lum is provided, it will match the mean and standard
        deviation of target_lum, where target_lum[0] is the mean and target_lum[1] is the standard deviation. If safe_values is enabled, it will
        find a target mean and standard deviation that is close to target_lum while not producing out-of-range values, i.e. outside of [0, 255].

            Warnings:
                - Clipping should be applied prior to uint8 conversion since np.uint8 and .astype('uint8') exhibit wrap-around behavior for out-of-range values. E.g. np.array([-2, 256]).astype('uint8') = [254, 0]
                - the target M and STD provided if safe_values is true, will not be equal to the grand average of the images' mean and std. Instead, it will find the closet mean and std that prevent out-of-range values.
        """

        def predict_values(original_means, original_stds, original_min_max, target_mean, target_std):
            """
            Predicts the minimum, maximum, and range values for a target distribution by transforming the
            original minimum and maximum values based on the provided target mean and standard deviation.
            The transformation is performed using the Z-score normalization formula.
            """
            predicted_min = (np.array(original_min_max)[:, 0] - np.array(original_means)) / np.array(original_stds) * target_std + target_mean
            predicted_max = (np.array(original_min_max)[:, 1] - np.array(original_means)) / np.array(original_stds) * target_std + target_mean
            predicted_range = predicted_max - predicted_min
            return predicted_min, predicted_max, predicted_range

        def compute_stats(im: np.ndarray, binary_mask: np.ndarray) -> Tuple[float, float, float, float]:
            """
            M, SD, min, max from RGB images whose channels are weighted (or not if grayscale).
            Args:
                im: An image
                binary_mask: A mask

            Returns:
                Tuple: mean, standard deviation
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
            #     ch_weights = RGB2GRAY_WEIGHTS[convertion_type]
            #     ch_means = np.array([np.mean(im[:, :, c][binary_mask[:, :, c]]) for c in range(3)])
            #     ch_stds = np.array([np.std(im[:, :, c][binary_mask[:, :, c]]) for c in range(3)])
            #     M = np.sum(ch_means * ch_weights)
            #     SD = np.sqrt(np.sum((ch_weights ** 2) * (ch_stds ** 2)))

            return M, SD, min, max

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
            M, SD, min, max = compute_stats(im=im, binary_mask=self.bool_masks[idx])
            original_means.append(M)
            original_stds.append(SD)
            original_min_max.append((min, max))
        target_mean, target_std = (np.mean(original_means), np.mean(original_stds)) if target_lum == (0, 0) else target_lum
        predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)

        if safe_values and (any(predicted_min < 0) or any(predicted_max > 255)):
            max_range = predicted_max.max() - predicted_min.min()
            scaling_factor = np.min([1, (255 - 1e-6) / max_range])  # Safety margin of 1e-6 to avoid precision issues
            target_std *= scaling_factor
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            target_mean = target_mean + (255 - np.max(predicted_max))
            console_log(msg=f"Adjusted target values for safe values: M = {target_mean:.4f}, SD = {target_std:.4f}", indent_level=0,color=Bcolors.WARNING, verbose=self.verbose > 2)
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            if np.any(predicted_min < -1e-3) or np.any(predicted_max > (255 + 1e-3)):
                raise Exception(f'Out-of-range values detected: mins = {list(predicted_min)}, maxs = {list(predicted_max)}')

        for idx, im in enumerate(buffer_collection):
            im2 = im3D(im.copy())
            M, SD, min, max = compute_stats(im=im2, binary_mask=self.bool_masks[idx])

            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            console_log(msg=f"Image {self._processed_image}", indent_level=0, color=Bcolors.BOLD, verbose=self.verbose>=2)
            console_log(msg=f"Original: M = {M:.4f}, SD = {SD:.4f}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose==3)

            # Standardization
            if original_stds[idx] != 0:
                im2[self.bool_masks[idx]] = (im2[self.bool_masks[idx]] - original_means[idx]) / original_stds[idx] * target_std + target_mean
            else:
                im2[self.bool_masks[idx]] = target_mean

            M, SD, min, max = compute_stats(im=im2, binary_mask=self.bool_masks[idx])

            # Save resulting image
            console_log(msg=f"Target values: M = {target_mean:.4f}, SD = {target_std:.4f}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose==3)
            self.dataset.buffer[idx] = im2  # update the dataset
            self._validate(observed=[M, SD], expected=[target_mean, target_std], measures_str=['M', 'SD'])

        self.dataset.buffer.drange = (0, 255)

    def hist_match(self):
        """
        Performs histogram matching on a collection of images to adjust their pixel intensities to match a
        target histogram. The method includes optional optimization steps to enhance the structural similarity
        index (SSIM) during the histogram matching process.

        Notes:
            - The input image collection, target histogram, and optimization options are managed as class-level
              attributes.
            - Includes safeguards to process images with different dynamic ranges.
            - Applies faster and deterministic exact_histogram_without_ties whenever possible.
            - Structural similarity and gradient maps are computed to guide the optimization process if enabled.

        Warnings:
            - Grayscale images: `hist_match` operates directly on intensity values and does not assume linear
              luminance scaling.
            - Color images: by default, `hist_match` is applied independently to each color channel.
              This may produce inaccurate color relationships or out-of-gamut results.
              If joint color consistency is required, consider using histogram matching of joint RGB distributions
              or other color-aware distribution matching methods.

        Raises:
            ValueError: If the target histogram's number of bins does not match the dynamic range of the processed
                        images.
        """

        hist_optim = self.options.hist_optim
        hist_spec_names = ['noise', 'moving-average', 'gaussian', 'hybrid', 'none']
        hist_specification = 5 if self.options.hist_specification is None else self.options.hist_specification
        tie_strategy = hist_spec_names[hist_specification-1]

        # Get appropriate image collection
        buffer_collection = self.dataset.buffer
        bit_size = 8
        n_bins = 2 ** bit_size

        # Get target histogram
        target_hist = self.options.target_hist
        if isinstance(target_hist, str):
            target_hist = np.ones((n_bins, buffer_collection.n_channels))/n_bins
        if target_hist is None:
            target_hist = avg_hist(buffer_collection, binary_masks=self.bool_masks, n_bins=n_bins)  # Placeholder for avgHist
        else:
            if target_hist.shape[0] != n_bins:
                raise ValueError(f"target_hist must have {n_bins} bins, but has {target_hist.shape[0]}.")
            if target_hist.max()>1:
                target_hist = target_hist.astype(np.float64)
                target_hist /= (target_hist.sum(axis=0, keepdims=True) + 1e-12)

        # If hist_optim disable, will run only one loop (n_iter = 1)
        n_iter = self.options.hist_iterations + 1 if hist_optim else 1  # See important note below to explain the +1. Also, note that the number of iterations for SSIM optimization (default = 10)
        step_sizes = self.options.step_size * np.ones(buffer_collection.n_channels)  # Step size (default = 34)
        original_step_sizes = step_sizes

        # Match the histogram
        self._processed_channel = None
        for idx, image in enumerate(buffer_collection):
            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            console_log(msg=f"Image {self._processed_image}", indent_level=0, color=Bcolors.BOLD, verbose=self.verbose>=2)

            image = im3D(image)
            X = image.copy()
            all_ssim = []
            ssim_increment = []
            for self._sub_iter in range(n_iter):  # n_iter = 1 when hist_optim == False
                if n_iter > 1 and self._sub_iter < n_iter - 1:
                    console_log(msg=f"SSIM optimization (iter={self._sub_iter + 1})", indent_level=1, color=Bcolors.BOLD, verbose=self.verbose >= 2)
                has_isoluminant_pixels = has_duplicates(X, binary_mask=self.bool_masks[idx])
                if tie_strategy != 'none' and has_isoluminant_pixels:
                    Y, OA = exact_histogram(image=X, binary_mask=self.bool_masks[idx], target_hist=target_hist, tie_strategy=tie_strategy, n_bins=n_bins)
                    if hist_spec_names != 'noise' and (n_iter == 1 or (n_iter > 1 and self._sub_iter < n_iter - 1)):
                        console_log(msg=f"Ordering accuracy per channel = {OA}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose == 3)
                else:  # If all values are unique, you don't need any tie-breaking solutions
                    if self._is_last_operation and self._sub_iter == (n_iter - 1) and has_isoluminant_pixels:
                        Y, OA = exact_histogram(image=X, binary_mask=self.bool_masks[idx], target_hist=target_hist, tie_strategy='hybrid', n_bins=n_bins)
                    else:
                        if not has_isoluminant_pixels:
                            console_log(msg=f"No ties detected: using direct histogram mapping", indent_level=1, color=Bcolors.OKCYAN, verbose=self.verbose == 3)
                        Y, _ = exact_histogram(image=X, binary_mask=self.bool_masks[idx], target_hist=target_hist, tie_strategy='none', n_bins=n_bins)

                # Compute Structural Similarity and gradient map (sens), along with max and min
                sens, ssim = ssim_sens(image, Y, data_range=n_bins-1)
                beta_bounds = beta_bounds_from_ssim(sens, ssim, self.bool_masks[idx])
                for ch in range(len(beta_bounds)):
                    min_beta, max_beta = beta_bounds[ch][0], beta_bounds[ch][1]
                    if min_beta < original_step_sizes[ch] > max_beta:
                        step_sizes[ch] = np.clip(original_step_sizes[ch], min_beta, max_beta)
                        console_log(msg=f"Original step size {original_step_sizes[ch]} is out of the optimal range for channel {ch}: [{min_beta}, {max_beta}].\nUsing step_size={step_sizes[ch]} instead.", indent_level=1, color=Bcolors.WARNING, verbose=self.verbose==3)

                # sens, ssim = ssim_sens(image/n_bins, Y/n_bins, n_bins=2)
                if n_iter > 1 and self._sub_iter < n_iter - 1:
                    all_ssim.append(ssim)
                    ssim_increment.append(np.mean(ssim))

                if hist_optim and (n_iter == 1 or (n_iter > 1 and self._sub_iter < n_iter - 1)):
                    console_log(msg=f"Mean SSIM = {np.mean(ssim):.4f}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose==3)
                if hist_optim and self._sub_iter < n_iter - 1:
                    ssim_update = np.zeros(sens.shape)
                    for ch in range(sens.shape[2]):
                        ssim_update[:, :, ch] = sens[:, :, ch] * step_sizes[ch] * self._sum_bool_masks[idx][ch]
                        X[:, :, ch] = Y[:, :, ch] + ssim_update[:, :, ch]  # X float64, Y uint8/uint16
                        # X = np.rint(np.clip(X, 0, 2 ** bit_size - 1)).astype(f'uint{bit_size}')

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
            rmse = compute_rmse(final_hist.flatten(), target_hist.flatten())
            console_log(msg=f"SSIM index between transformed and original image: {np.mean(ssim):.4f}", indent_level=1, color=Bcolors.OKBLUE, verbose=self.verbose==3)
            self._validate(observed=[rmse], expected=[0], measures_str=['RMS error'])

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
            - output values are typically out-of-range. This does not represent a problem
                for iterative modes [5, 7] as hist_match could benefit from images without
                duplicated values. Values need to be readjusted when
                self._is_last_operation is True.
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
        f_cols = np.fft.fftshift(np.fft.fftfreq(y_size, d=1 / y_size))  # like f1 in MATLAB
        f_rows = np.fft.fftshift(np.fft.fftfreq(x_size, d=1 / x_size))  # like f2 in MATLAB
        XX, YY = np.meshgrid(f_cols, f_rows)
        nyquistLimit = np.floor(max(x_size, y_size) / 2)
        r, theta = cart2pol(XX, YY)

        # Map of the frequency bins.
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
            console_log(msg=f"Image {self._processed_image}", indent_level=0, color=Bcolors.BOLD, verbose=self.verbose>=2)
            matched_image = []
            magnitude = im3D(self.dataset.magnitudes[idx])
            phase = im3D(self.dataset.phases[idx])
            for self._processed_channel in range(n_channels):
                console_log(msg=f'Channel {self._processed_channel}', indent_level=1, verbose=self.verbose>=2, color=Bcolors.BOLD)
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

                output = np.real(np.fft.ifft2(np.fft.ifftshift(new)))
                matched_image.append(output)

                # Comparison: obtained vs target rotational averages (up to Nyquist)
                obtained_ra = rot_avg(new_magnitude, radius=r1)
                R = int(min(len(obtained_ra), len(target_ra), nyquistLimit + 1))
                t = target_ra[:R]
                o = obtained_ra[:R]
                rmse = compute_rmse(t, o)
                self._validate(observed=[rmse], expected=[0],
                               measures_str=['RMS error'])

            # Soft-clip output values: As this transformation typically produces out-of-range values
            output_image = np.stack(matched_image, axis=-1).squeeze()
            if self._is_last_operation:
                mn, mx = output_image.min(), output_image.max()
                if 0 < mn > 1:
                    console_log(
                        msg=f'Out of range values: Actual range [{mn}, {mx}] outside of the admitted range [0, 1].\nWill be rescaled and clipped so that less than 1% falls outside of [0, 1].',
                        indent_level=1, color=Bcolors.WARNING, verbose=self.verbose==3)
                    output_image = soft_clip(output_image, min_value=0, max_value=1, max_percent=0.01, verbose=self.verbose==3)
            buffer_collection[idx] = output_image * 255

        buffer_collection.drange = (0, 255)

        # buffer_collection dtype is np.float64 and drange is close but out of [0, 1] before rescaling of any sort
        # TODO: NEEDS TO BE CHECKED
        if self.options.rescaling and self._is_last_operation:
            buffer_collection = rescale_images255(buffer_collection, rescaling_option=self.options.rescaling)
            # If legacy mode is turned on, rescale_images255 will output uint8
            if not np.issubdtype(buffer_collection.dtype, np.floating):
                for idx, image in enumerate(buffer_collection):
                    buffer_collection[idx] = image.astype('float64')
                buffer_collection.drange = (0, 255)

    def spec_match(self):
        """Match the full magnitude spectrum of images to a target spectrum.

        This function reconstructs images whose Fourier magnitude is replaced
        by the `target_spectrum`, while preserving the original Fourier phase.
        The inverse FFT is then used to get spatial-domain images with the
        desired spectral characteristics.

        Notes:
            - Phase information from each input image is preserved.
            - The output is real-valued because only magnitude is replaced.
            - output values are typically out-of-range. This does not represent a problem
            for iterative modes [6, 8] as hist_match could benefit from images without
            duplicated values. Values need to be readjusted when self._is_last_operation
            is True.
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

        # Ensure the target spectrum is 3D (H, W, C)
        target_spectrum = im3D(target_spectrum)
        x_size, y_size, n_channels = target_spectrum.shape[:3]
        image = im3D(self.dataset.images[0])
        if target_spectrum.shape != image.shape:
            raise TypeError('The target spectrum must have the same size as the images.')

        # Iterate over each image (each entry in the phase collection)
        for idx, image in enumerate(buffer_collection):
            self._processed_image = f'#{idx}' if self.dataset.images.src_paths[idx] is None else self.dataset.images.src_paths[idx]
            console_log(msg=f"Image {self._processed_image}", indent_level=0, color=Bcolors.BOLD, verbose=self.verbose>=2)

            matched_image = []

            # Convert the stored phase to 3D array
            phase = im3D(self.dataset.phases[idx])

            # Process each channel separately
            for self._processed_channel in range(n_channels):
                console_log(msg=f"Channel {self._processed_channel}", indent_level=1, color=Bcolors.BOLD, verbose=self.verbose>=2)

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

                rmse = compute_rmse(t, o)
                self._validate(observed=[rmse], expected=[0],
                               measures_str=['RMS error'])

            # Soft-clip output values: As this transformation typically produces out-of-range values
            output_image = np.stack(matched_image, axis=-1).squeeze()
            if self._is_last_operation:
                mn, mx = output_image.min(), output_image.max()
                if 0 < mn > 1:
                    console_log(
                        msg=f'Out of range values: Actual range [{mn}, {mx}] outside of the admitted range [0, 1].\nWill be rescaled and clipped so that less than 1% falls outside of [0, 1].',
                        indent_level=1, color=Bcolors.WARNING, verbose=self.verbose==3)
                    output_image = soft_clip(output_image, min_value=0, max_value=1, max_percent=0.01, verbose=self.verbose == 3)

            # Stack the channels and save into the output collection
            buffer_collection[idx] = output_image * 255

        buffer_collection.drange = (0, 255)
        # buffer_collection dtype is np.float64 and drange is close but out of [0, 1] before rescaling of any sort
        # TODO: NEEDS TO BE CHECKED
        if self.options.rescaling and self._is_last_operation:
            buffer_collection = rescale_images255(buffer_collection, rescaling_option=self.options.rescaling)
            # If legacy mode is turned on, rescale_images255 will output uint8
            if not np.issubdtype(buffer_collection.dtype, np.floating):
                for idx, image in enumerate(buffer_collection):
                    buffer_collection[idx] = image.astype(float)
                buffer_collection.drange = (0, 255)
