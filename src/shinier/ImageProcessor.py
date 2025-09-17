from typing import Optional, List, Union, Iterable, Tuple
Vector = Iterable[Union[float, int]]
from datetime import datetime
import inspect
import numpy as np

from shinier import ImageDataset, Options
from shinier.utils import (
    ImageListType, separate, imhist, im3D,
    rescale_images, get_images_spectra, ssim_sens, cart2pol,
    pol2cart, float01_to_uint, uint_to_float01, noisy_bit_dithering,
    exact_histogram, exact_histogram_with_noise, Bcolors, MatlabOperators, compute_rmse)


class ImageProcessor:
    """Base class for image processing."""
    def __init__(self, dataset: ImageDataset, options: Optional[Options] = None, verbose: bool = True):
        self.dataset: ImageDataset = dataset
        self.options: Optional[Options] = dataset.options or getattr(dataset, "options", None)
        self.current_image: Optional[np.ndarray] = None
        self.current_masks: Optional[np.ndarray] = None
        self.bool_masks: List = [None] * len(self.dataset.images)
        self.verbose: bool = verbose
        self.log: List = []
        self.seed: int = None
        self._state_dict = {
            1: {'lum_match': ['images', 'images']},
            2: {'hist_match': ['images', 'images']},
            3: {'fourier_match': ['images', 'images']},
            4: {'fourier_match': ['images', 'images']},
            5: {'hist_match': ['images', 'buffer'], 'fourier_match': ['buffer', 'images']},
            6: {'hist_match': ['images', 'buffer'], 'fourier_match': ['buffer', 'images']},
            7: {'fourier_match': ['images', 'buffer'], 'hist_match': ['buffer', 'images']},
            8: {'fourier_match': ['images', 'buffer'], 'hist_match': ['buffer', 'images']},
            9: {'only_dithering': ['images', 'images']}
        }
        self.seed = self.options.seed
        self.process()
        self.dataset.print_log()
        # self.computed_metrics = compute_metrics_from_paths(self.dataset, self.options)
        if not self.dataset.images.has_list_array:
            self.dataset.save_images()
            self.dataset.close()
        else:
            print(f'{Bcolors.WARNING}To get the output images, you must instantiate ImageProcessor and call get_results() method. \n\tE.g.: output_images = ImageProcessor(dataset=my_dataset).get_results(){Bcolors.ENDC}')

    def get_results(self):
        """Return list of processed np.ndarray if input was arrays, otherwise None."""
        if getattr(self.dataset.images, "file_paths", [None])[0] is None:
            return self.dataset.images.data
        else:
            return

    def _get_relevant_input_output(self):
        # Get the caller from the stack
        self._state_function = inspect.stack()[1].function # The caller is at index 1 (0 is the current method itself)
        input_name, output_name = self._state_dict[self.options.mode][self._state_function]
        # print(f"Method was called by {self._state_function}\nInput name: {input_name}\nOutput name: {output_name}") if self.verbose else None
        input_collection = getattr(self.dataset, input_name)
        output_collection = getattr(self.dataset, output_name)
        return input_collection, input_name, output_collection, output_name

    def _apply_post_processing(self, output_name: str, output_collection: ImageListType, dithering: bool):
        # Applies dithering if required and last processing step
        if output_name == 'images': # Apply post_processing only if last processing step
            dtype = output_collection.dtype
            drange = output_collection.drange
            if dithering: #Make sure input images are float01
                for idx, image in enumerate(output_collection):
                    # Dithering function assumes float with values in the [0, 1] range
                    if not np.issubdtype(dtype, np.floating):
                        image = uint_to_float01(image)
                    elif np.issubdtype(dtype, np.floating) and drange != (0, 1):
                        image = image/max(drange)
                    output_collection[idx] = noisy_bit_dithering(image, 256)
                dtype = output_collection.dtype
                drange = output_collection.drange

            # Make sure the output is uint8 with proper range
            for idx, image in enumerate(output_collection):
                if drange != (0, 255):
                    image = MatlabOperators.uint8(image/max(drange)*255) if self.options.legacy_mode else np.clip(image/max(drange)*255, 0, 255).astype(np.uint8)
                elif dtype != np.uint8:
                    image = MatlabOperators.uint8(image) if self.options.legacy_mode else image.astype(np.uint8)
                output_collection[idx] = image.squeeze()

        return output_collection

    def _set_relevant_output(self, output_collection: ImageListType, output_name: str):
        if output_name == 'images':
            output_collection.file_paths = self.dataset.images.file_paths # If final output, make sure the file_paths are aligned with input file_paths.
        setattr(self.dataset, output_name, output_collection)

    def _get_mask(self, idx):
        """ Provide mask if masks exists in the dataset, if not make blank masks (all True). """
        n_dims = self.dataset.images.n_dims
        im_size = self.dataset.images.reference_size
        if self.bool_masks[idx] is None:
            if self.options.whole_image == 2:
                print('prepare mask (whole-image 2)') if self.verbose else None
                self._prepare_mask(image = self.dataset.images[idx])
                self.bool_masks[idx] = np.stack((self.current_masks[0],) * (3 if n_dims == 3 else 1), axis=-1).squeeze()
            elif self.options.whole_image == 3:
                if idx < self.dataset.n_masks: # If there is one mask, it picks self.bool_masks[0] everytime
                    print('prepare mask (whole-image : 3)') if self.verbose else None
                    self._prepare_mask(image = self.dataset.images[idx], mask = self.dataset.masks[idx])
                    self.bool_masks[idx] = np.stack((self.current_masks[0],) * (3 if n_dims == 3 else 1), axis=-1).squeeze()
                else:
                    self.bool_masks[idx] = self.bool_masks[0]
            else:
                masks_size = im_size
                if n_dims == 3:
                    masks_size = list(im_size) + [n_dims]
                self.bool_masks[idx] = np.ones(masks_size, dtype=bool).squeeze() if idx == 0 else self.bool_masks[0]

    def _prepare_mask(self, image, mask = None):
        if self.options.whole_image == 2:
            mask_f, mask_b, _ = separate(image, self.options.background)
        elif self.options.whole_image == 3:
            mask_f, mask_b, _ = separate(mask, self.options.background)
        self.current_masks = (mask_f, mask_b)

    def process(self):
        if self.options.mode in [2, 5, 6, 7, 8] and self.options.hist_specification:
            # Set a seed for the random generator used in exact histogram specification
            if self.seed is None:
                now = datetime.now()
                self.seed = int(now.timestamp())
            np.random.seed(self.seed)
            self.dataset.processing_logs.append(f'seed={self.seed}')
            print(f'{Bcolors.WARNING}Use this seed for reproducibility: {self.seed}{Bcolors.ENDC}')
        if self.options.mode == 1:
            print(f'{Bcolors.OKGREEN}Applying luminance matching...{Bcolors.ENDC}')
            self.dataset.processing_logs.append('lum_match')
            self.lum_match(target_lum=self.options.target_lum, safe_values=self.options.safe_lum_match)
        if self.options.mode in [2, 5, 6]:
            print(f'{Bcolors.OKGREEN}Applying histogram matching...{Bcolors.ENDC}')
            self.dataset.processing_logs.append('hist_match')
            self.hist_match(target_hist=self.options.target_hist, hist_optim=self.options.hist_optim, hist_specification=self.options.hist_specification)
        if self.options.mode in [3, 5, 7]:
            print(f'{Bcolors.OKGREEN}Applying spatial frequency matching...{Bcolors.ENDC}')
            self.dataset.processing_logs.append('sf_match')
            self.fourier_match(target_spectrum=self.options.target_spectrum, rescaling_option=self.options.rescaling, matching_type='sf')
        if self.options.mode in [4, 6, 8]:
            print(f'{Bcolors.OKGREEN}Applying spectrum matching...{Bcolors.ENDC}')
            self.dataset.processing_logs.append('spec_match')
            self.fourier_match(target_spectrum=self.options.target_spectrum, rescaling_option=self.options.rescaling, matching_type='spec')
        if self.options.mode in [7, 8]:
            print(f'{Bcolors.OKGREEN}Applying histogram matching...{Bcolors.ENDC}')
            self.dataset.processing_logs.append('hist_match')
            self.hist_match(target_hist=self.options.target_hist, hist_optim=self.options.hist_optim, hist_specification=self.options.hist_specification)
        if self.options.mode == 9:
            print(f'{Bcolors.OKGREEN}Applying dithering only...{Bcolors.ENDC}')
            print('Applying dithering...')
            self.only_dithering()
            self.dataset.processing_logs.append('only_dithering')

    def only_dithering(self):
        """ Applies the noisy-bit dithering to the input images. """
        input_collection, input_name, output_collection, output_name = self._get_relevant_input_output()

        buffer_collection = self._apply_post_processing(output_name, output_collection, dithering=True)
        self._set_relevant_output(buffer_collection, output_name)

    def lum_match(self, target_lum: Optional[Iterable[Union[float, int]]] = (0, 0), safe_values: bool = False):
        """
        Matches the mean and standard deviation of a set of images. If target_lum is provided, it will match the mean and standard
        deviation of target_lum, where target_lum[0] is the mean and target_lum[1] is the standard deviation. If safe_values is enabled, it will
        find a target mean and standard deviation that is close to target_lum while not producing out-of-range values, i.e. outside of [0, 255].

            Args:
                images (ImageListType): A list of grayscale images to be processed.
                masks (ImageListType): Optional. A list of mask(s) for figure-ground segregation.
                    Each mask contains ones where the histograms are obtained (e.g., foreground) and zeros elsewhere.
                target_lum (Iterable[Union[float, int]]): Optional. An iterable of the requested mean and standard deviation.
                safe_values (bool): If true, the mean and standard deviation used to match the images will be computed so that all resulting values remain in the [0, 255] range.

            Warnings:
                - Clipping should be applied prior to uint8 conversion since np.uint8 and .astype('uint8') exhibit wrap-around behavior for out-of-range values. E.g. np.array([-2, 256]).astype('uint8') = [254, 0]
                - the target M and STD provided if safe_values is true, will not be equal to the grand average of the images' mean and std. Instead, it will find the closet mean and std that prevent out-of-range values.
        """
        def predict_values(original_means, original_stds, original_min_max, target_mean, target_std):
            predicted_min = (np.array(original_min_max)[:, 0] - np.array(original_means))/np.array(original_stds) * target_std + target_mean
            predicted_max = (np.array(original_min_max)[:, 1] - np.array(original_means))/np.array(original_stds) * target_std + target_mean
            predicted_range = predicted_max - predicted_min
            return predicted_min, predicted_max, predicted_range

        # 1) Compute the mean and standard deviation of the original images.
        # 2) Compute the target mean and standard deviation if not provided.
        # 3) Adjust the target mean and standard deviation if safe_values is enabled and if there are out-of-range values.
        # 4) Convert images into float
        # 5) Rescale according to target mean and standard deviation
        # 6) Apply clipping if needed
        # 7) Convert images back into uint8
        original_means, original_stds, original_min_max = [], [], []
        for idx, im in enumerate(self.dataset.images):
            self._get_mask(idx)
            M = MatlabOperators.mean2(im[self.bool_masks[idx]]) if self.options.legacy_mode else np.mean(im[self.bool_masks[idx]])
            SD = MatlabOperators.std2(im[self.bool_masks[idx]]) if self.options.legacy_mode else np.std(im[self.bool_masks[idx]])
            original_means.append(M)
            original_stds.append(SD)
            original_min_max.append((im[self.bool_masks[idx]].min(), im[self.bool_masks[idx]].max()))
        target_mean, target_std = (np.mean(original_means), np.mean(original_stds)) if target_lum == (0, 0) else target_lum
        predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)

        if safe_values and (any(predicted_min<0) or any(predicted_max>255)):
            max_range = predicted_max.max() - predicted_min.min()
            scaling_factor = min(1, (255 - 1e-6) / max_range)  # Safety margin of 1e-6 to avoid precision issues
            target_std *= scaling_factor
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            target_mean = target_mean + (255 - np.max(predicted_max))
            print(f"{Bcolors.OKBLUE}Adjusted target values for safe values: M = {target_mean:.4f}, SD = {target_std:.4f}{Bcolors.ENDC}") if self.verbose else None
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            if (any(predicted_min < 0) or any(predicted_max > 255)):
                raise Exception(f'Out-of-range values detected: mins = {list(predicted_min)}, maxs = {list(predicted_max)}')

        for idx, im in enumerate(self.dataset.images):
            im2 = im.copy().astype(float)
            M = MatlabOperators.mean2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.mean(im2[self.bool_masks[idx]])
            SD = MatlabOperators.std2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.std(im2[self.bool_masks[idx]])
            im_name = f'#{idx}' if self.dataset.images.file_paths[idx] is None else self.dataset.images.file_paths[idx]
            print(f'{Bcolors.OKGREEN}Image {im_name}:{Bcolors.ENDC}')
            print(f'{Bcolors.OKBLUE}\tOriginal:\t\t\t\tM = {M:.4f}, SD = {SD:.4f}{Bcolors.ENDC}') if self.verbose else None

            # Standardization
            if original_stds[idx] != 0:
                im2[self.bool_masks[idx]] = (im2[self.bool_masks[idx]] - original_means[idx]) / original_stds[idx] * target_std + target_mean
            else:
                im2[self.bool_masks[idx]] = target_mean

            M = MatlabOperators.mean2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.mean(im2[self.bool_masks[idx]])
            SD = MatlabOperators.std2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.std(im2[self.bool_masks[idx]])
            # print(f'\t{Bcolors.OKBLUE}Standardized (float):\tM = {M:.4f}, SD = {SD:.4f}{Bcolors.ENDC}') if self.verbose else None
            mx, mn = np.max(im2[self.bool_masks[idx]]), np.min(im2[self.bool_masks[idx]])
            clipping_needed = mn<0 or mx>255
            print(f"{Bcolors.WARNING}Warning: Clipping applied because values of image #{idx} are outside the [0, 255] range: [{mn}, {mx}]. Results of lum_match might not be exact{Bcolors.ENDC}") if self.verbose and clipping_needed else None
            im2 = MatlabOperators.uint8(im2) if self.options.legacy_mode else np.clip(im2, 0, 255).astype('uint8')

            # Save resulting image
            M = MatlabOperators.mean2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.mean(im2[self.bool_masks[idx]])
            SD = MatlabOperators.std2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.std(im2[self.bool_masks[idx]])
            print(f'\t{Bcolors.OKBLUE}Standardized (uint8):\tM = {M:.4f}, SD = {SD:.4f}{Bcolors.ENDC}') if self.verbose else None
            print(f'\t{Bcolors.OKBLUE}Target values:\t\t\tM = {target_mean:.4f}, SD = {target_std:.4f}{Bcolors.ENDC}') if self.verbose else None
            if M != target_mean or SD != target_std:
                print(f'\t{Bcolors.WARNING}* Discrepancies between Target and Standardized values are due to the conversion of floats into integers{Bcolors.ENDC}') if self.verbose else None
            print('\n')
            self.dataset.images[idx] = im2 #update the dataset
            self.dataset.images.drange = (0, 255)
            self.dataset.processing_logs.append(f'Image {im_name}:\n\tObserved: M = {M:.4f}, SD = {SD:.4f}\n\tExpected: M = {target_mean:.4f}, SD = {target_std:.4f}')

        self.dataset.images = self._apply_post_processing(
            output_name='images',
            output_collection=self.dataset.images,
            dithering=self.options.dithering)

    def hist_match(self, target_hist: Optional[np.ndarray] = None, hist_optim: Optional[int] = 1, hist_specification: Optional[int] = 0, noise_level=0.1, n_bins: int = 256):
        """
        Equates a set of images in terms of luminance histograms.

        Args:
            target_hist (Optional[np.ndarray]): Target histogram. If not provided, it's computed using _avg_hist().
            hist_specification (Optional[int]): If 0, applies exact histogram specification without noise (see Coltuc, Bolon & Chassery, 2006)
                If 1, uses noise for exact histogram specification (legacy code).
            noise_level (float): Noise level to be applied in case hist_specification is set to 1. Default is 0.1.
            hist_optim (Optional[int]): If 0, only basic histogram matching is performed (default).
                If 1, uses the SSIM optimization method of Avanaki (2009).
            n_bins (int): Number of gray levels (default is 256).
        """

        def _get_hist(image: np.ndarray, mask: np.ndarray, n_bins=256, normalized: bool = False):
            hist_data = imhist(image, self.bool_masks[idx], n_bins=n_bins)
            if normalized:
                hist_data = hist_data/hist_data.sum()
            return hist_data

        def _avg_hist(images: ImageListType, normalized: bool=True) -> np.ndarray:
            n_channels = 1 if images.n_dims == 2 else 3
            n_bins = max(images.drange) + 1
            hist_sum = np.zeros((n_bins, n_channels))
            for idx, im in enumerate(images):
                self._get_mask(idx) # Prepare the mask if it does not already exist. Default mask is all True.
                hist_sum += imhist(im, self.bool_masks[idx], n_bins=n_bins)

            # Average of the pixels in the bins
            average = hist_sum / len(images)
            if normalized:
                average /= average.sum()

            return average

        # Get appropriate image collection
        input_collection, input_name, output_collection, output_name = self._get_relevant_input_output()
        buffer_collection = self.dataset.buffer
        dtype = input_collection.dtype
        drange = input_collection.drange

        # Adjust bit-size if needed: This is to ensure maximum bit size for discrete operation (see bit_size = 16 section below)
        bit_size = 8
        to_convert = False
        if (np.issubdtype(dtype, np.floating) and self.options.mode in [7, 8]) or (hist_specification == 0 and hist_optim == 1):
            # Convert float to uint16 to preserve as much info as possible during exact hist_matching which relies on integers.
            bit_size = 16
            to_convert = True
        n_bins = 2 ** bit_size

        for idx, image in enumerate(input_collection):
            if to_convert:
                if np.issubdtype(dtype, np.floating):
                    if drange == (0, 1):
                        buffer_collection[idx] = float01_to_uint(image, allow_clipping=True, bit_size=bit_size)
                    else:
                        print(f'{Bcolors.WARNING}Why would this happen?{Bcolors.ENDC}')

                elif dtype == np.uint8:
                    buffer_collection[idx] = (image / 255.0 * (2 ** bit_size - 1)).astype(np.dtype(f'uint{bit_size}'))
            else:
                buffer_collection[idx] = image

        if target_hist is None:
            target_hist = _avg_hist(buffer_collection)  # Placeholder for avgHist
        else:
            for idx, im in enumerate(buffer_collection):
                self._get_mask(idx)
            if target_hist.shape[0] != n_bins:
                raise ValueError(f"target_hist must have {n_bins} bins, but has {target_hist.shape[0]}.")

        # If hist_optim disable, will run only one loop (n_iter = 1)
        n_iter = self.options.iterations if hist_optim else 1  # Number of iterations for SSIM optimization (default = 10)
        step_size = self.options.step_size  # Step size (default = 34)

        # Match the histogram
        for idx, image in enumerate(buffer_collection):
            im_name = f'#{idx}' if self.dataset.images.file_paths[idx] is None else self.dataset.images.file_paths[idx]
            print(f'{Bcolors.OKGREEN}Image {im_name}:{Bcolors.ENDC}')

            image = im3D(image)
            X = image
            M = np.prod(image.shape)/image.shape[2]
            for iter in range(n_iter): # n_iter = 1 when hist_optim == 0
                print(f'{Bcolors.OKGREEN}\tIteration #{iter + 1}: {Bcolors.ENDC}') if self.verbose and n_iter > 1 else None

                if hist_specification:
                    Y = exact_histogram_with_noise(image=X, binary_mask=self.bool_masks[idx], target_hist=target_hist, noise_level=noise_level, n_bins=n_bins)
                else:
                    Y, OA = exact_histogram(image=X, target_hist=target_hist, binary_mask=self.bool_masks[idx], verbose=self.verbose)
                    print(f'{Bcolors.OKBLUE}\ttOrdering accuracy per channel = {OA}{Bcolors.ENDC}') if self.verbose else None

                sens, ssim = ssim_sens(image, Y, n_bins=n_bins)
                print(f'{Bcolors.OKBLUE}\t\tMean SSIM = {np.mean(ssim):.4f}{Bcolors.ENDC}') if self.verbose and hist_optim else None
                if hist_optim:
                    ssim_update = sens * step_size * M
                    X = Y + ssim_update # X float64, Y uint8/uint16
                    X = np.clip(X, 0, 2**bit_size - 1).astype(np.uint16) if bit_size == 16 else np.clip(X, 0, 2**bit_size - 1)
                    if iter == n_iter-1:
                        new_image = X.astype(np.uint16) if bit_size == 16 else X.astype(np.uint8)
                else:
                    new_image = Y.astype(np.uint16) if bit_size == 16 else Y.astype(np.uint8)

            buffer_collection[idx] = new_image

            # Compute statistics
            final_hist = _get_hist(new_image, mask=self.bool_masks[idx], n_bins=n_bins, normalized=True)
            corr = np.corrcoef(final_hist.flatten(), target_hist.flatten())
            rmse = compute_rmse(final_hist.flatten(), target_hist.flatten())
            print(f'{Bcolors.OKBLUE}\tCorrelation between transformed and target histogram: {corr[0, 1]:.4f}{Bcolors.ENDC}') if self.verbose else None
            print(f'{Bcolors.OKBLUE}\tRMS error between transformed and target histogram: {rmse:.4f}{Bcolors.ENDC}') if self.verbose else None
            print(f'{Bcolors.OKBLUE}\tSSIM index between transformed and original image: {np.mean(ssim):.4f}{Bcolors.ENDC}') if self.verbose else None
            self.dataset.processing_logs.append(f'Image {im_name}:\n\tCorrelation between transformed and target histogram: {corr[0, 1]:.4f}\n\tRMS error between transformed and target histogram: {rmse:.4f}\n\tSSIM index between transformed and original image: {np.mean(ssim):.4f}')


        output_collection = self._apply_post_processing(output_name=output_name, output_collection=buffer_collection, dithering=self.options.dithering)
        self._set_relevant_output(output_collection, output_name)

    def fourier_match(self, target_spectrum: Optional[np.ndarray] = None, matching_type: str = 'sf', rescaling_option: Optional[int] = 1) -> List[np.ndarray]:
        """
        Match either the rotational average of the Fourier amplitude or the entire spectrum for a set of images.

        Args:
            target_spectrum (Optional[np.ndarray]) : Target magnitude spectrum. Same size as the images.
                If None, the target magnitude spectrum is the average spectrum of the all the input images.
                    E.g.,
                        fftim = np.fft.fftshift(np.fft.fft2(im))
                        rho, theta = self.cart2pol(np.real(fftim), np.imag(fftim))
                        tarmag = rho
            matching_type (str): if "sf", it will match the rotational average of the fourier. If "spec" it will
                match the entire spectrum.
            rescaling_option (Optional[int]) : Determines whether the luminance values are rescaled after the image modification
                0 : Rescaling self max/min
                1 : Rescaling absolute max/min (Default)
                2 : Rescaling average max/min

        Returns :
            List of images matched on the rotational average.
        """

        def _sf_match(input_collection: ImageListType, output_collection: ImageListType, magnitudes: ImageListType, phases: ImageListType, target_spectrum: np.ndarray):
            target_spectrum = im3D(target_spectrum)
            x_size, y_size, n_channels = target_spectrum.shape[:3]

            #  Returns the frequencies of the image, bins range from -0.5f to 0.5f (0.5f is the Nyquist frequency) 1/y_size is the distance between each pixel in the image
            f1 = np.fft.fftshift(np.fft.fftfreq(x_size, d=1 / x_size))
            f2 = np.fft.fftshift(np.fft.fftfreq(y_size, d=1 / y_size))
            nyquistLimit = np.floor(max(x_size, y_size) / 2)
            XX, YY = np.meshgrid(f1, f2)
            r, theta = cart2pol(XX, YY)

            # Map of the bins of the frequencies
            r = MatlabOperators.round(r) if self.options.legacy_mode else np.round(r, decimals=0)

            # Need to be a 1D array of integers for the bincount function
            r1 = r.flatten().astype(np.uint16)

            # Match spatial frequency on rotational average of the magnitude spectrum
            for idx in range(len(input_collection)):
                matched_image = []
                magnitude = im3D(magnitudes[idx])
                phase = im3D(phases[idx])
                for channel in range(n_channels):
                    fft_image = magnitude[:, :, channel]
                    source_rotational_avg = np.bincount(r1, weights=fft_image.flatten())
                    target_rotational_avg = np.bincount(r1, weights=target_spectrum[:, :, channel].flatten())
                    coefficient = target_rotational_avg / source_rotational_avg

                    # For where in r the value is j, apply the coefficient of index j to cmat
                    cmat = np.zeros_like(r)
                    for j in range(len(coefficient)):
                        cmat[r == j] = coefficient[j]

                    # Remove frequencies higher than the Nyquist frequency
                    cmat[r > nyquistLimit] = 0

                    # Compute new magnitude and convert back to image
                    new_magnitude = fft_image * cmat

                    [XX, YY] = pol2cart(new_magnitude, phase[:, :, channel])
                    new = XX + YY * 1j  # 1j = sqrt(-1)
                    output = np.real(np.fft.ifft2(np.fft.ifftshift(new)))
                    matched_image.append(output)
                output_collection[idx] = np.stack(matched_image, axis=-1).squeeze()
            return output_collection

        def _spec_match(output_collection: ImageListType, phases : ImageListType, target_spectrum: np.ndarray):
            target_spectrum = im3D(target_spectrum)
            x_size, y_size, n_channels = target_spectrum.shape[:3]

            # Match spatial frequency on rotational average of the magnitude spectrum
            for idx in range(len(phases)):
                matched_image = []
                phase = im3D(phases[idx])
                for channel in range(n_channels):
                    XX, YY = pol2cart(target_spectrum[:, :, channel], phase[:, :, channel])
                    new = XX + YY * 1j  # 1j = sqrt(-1)
                    output = np.real(np.fft.ifft2(np.fft.ifftshift(new)))
                    matched_image.append(output)
                output_collection[idx] = np.stack(matched_image, axis=-1).squeeze()
            return output_collection

        # Get proper input and output image collections
        input_collection, input_name, output_collection, output_name = self._get_relevant_input_output()
        buffer_collection = self.dataset.buffer

        # Compute all spectra
        if self.dataset.magnitudes.dtype == np.bool:
            self.dataset.magnitudes, self.dataset.phases = get_images_spectra(images=input_collection, magnitudes=self.dataset.magnitudes, phases=self.dataset.phases)

        # If target_spectrum is None, target magnitude is the rotational average of the spectrum
        if target_spectrum is None:
            target_spectrum = np.zeros(self.dataset.magnitudes[0].shape)
            for idx, mag in enumerate(self.dataset.magnitudes):
                target_spectrum += mag
            target_spectrum /= len(self.dataset.magnitudes)
        else:
            if target_spectrum.shape != self.dataset.images.reference_size:
                raise TypeError('The target spectrum must have the same size as the images.')

        # Apply the relevant Fourier match
        if matching_type == 'sf':
            buffer_collection = _sf_match(input_collection=input_collection, output_collection=buffer_collection, magnitudes=self.dataset.magnitudes, phases=self.dataset.phases, target_spectrum=target_spectrum)
        elif matching_type == 'spec':
            buffer_collection = _spec_match(output_collection=buffer_collection, phases=self.dataset.phases, target_spectrum=target_spectrum)

        # buffer_collection dtype is np.float64 and drange is close but out of [0, 1] before rescaling of any sort
        if self.options.rescaling:
            buffer_collection = rescale_images(buffer_collection, rescaling_option=self.options.rescaling)
        else :
            for idx in range(len(buffer_collection)):
                buffer_collection[idx] = (np.clip(buffer_collection[idx], 0, 1) * 255).astype(np.uint8)

        buffer_collection = self._apply_post_processing(output_name, buffer_collection, dithering=self.options.dithering)
        self._set_relevant_output(buffer_collection, output_name)

