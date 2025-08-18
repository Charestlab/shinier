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
    exact_histogram, compute_metrics_from_paths, bcolors, MatlabOperators)


class ImageProcessor:
    """Base class for image processing."""
    def __init__(self, dataset: ImageDataset, verbose: bool=True):
        self.dataset: ImageDataset=dataset
        self.options: Options=dataset.options
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
            8: {'fourier_match': ['images', 'buffer'], 'hist_match': ['buffer', 'images']}
        }
        self.seed = self.options.seed
        self.process()
        self.computed_metrics = compute_metrics_from_paths(self.dataset, self.options)
        self.dataset.save_images()
        self.dataset.print_log()
        self.dataset.close()

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
                    output_collection[idx] = noisy_bit_dithering(image)
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
        """
        Provide mask if masks exists in the dataset, if not make a blank masks (all True).
        Make sure there are the same number of masks than images and that they are of the same size.
        """
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
        if self.options.mode not in range(1, 9):
            raise ValueError('Options.mode should be between 1 and 8')

        if self.options.mode in [2, 5, 6, 7, 8] and self.options.hist_specification:
            # Set a seed for the random generator used in exact histogram specification
            if self.seed is None:
                now = datetime.now()
                self.seed = int(now.timestamp())
            np.random.seed(self.seed)
            self.dataset.processing_steps.append(f'seed={self.seed}')
            print(f'Use this seed for reproducibility: {self.seed}')
        if self.options.mode == 1:
            print('Applying luminance matching...')
            self.lum_match(lum=self.options.target_lum, safe_values=self.options.safe_lum_match)
            self.dataset.processing_steps.append('lum_match')
        if self.options.mode in [2, 5, 6]:
            print('Applying histogram matching...')
            self.hist_match(target_hist=self.options.target_hist, hist_optim=self.options.hist_optim, hist_specification=self.options.hist_specification)
            self.dataset.processing_steps.append('hist_match')
        if self.options.mode in [3, 5, 7]:
            print('Applying spatial frequency matching...')
            self.fourier_match(target_spectrum = self.options.target_spectrum, rescaling_option=self.options.rescaling, matching_type='sf')
            self.dataset.processing_steps.append('sf_match')
        if self.options.mode in [4, 6, 8]:
            print('Applying spectrum matching...')
            self.fourier_match(target_spectrum = self.options.target_spectrum, rescaling_option=self.options.rescaling, matching_type='spec')
            self.dataset.processing_steps.append('spec_match')
        if self.options.mode in [7, 8]:
            print('Applying histogram matching...')
            self.hist_match(target_hist=self.options.target_hist, hist_optim=self.options.hist_optim, hist_specification=self.options.hist_specification)
            self.dataset.processing_steps.append('hist_match')

    def lum_match(self, lum: Optional[Iterable[Union[float, int]]] = (0, 0), safe_values: bool = False):
        """
        Matches the mean and standard deviation of a set of images. If lum is provided, it will match the mean and standard
        deviation of lum, where lum[0] is the mean and lum[1] is the standard deviation. If safe_values is enabled, it will
        find a target mean and standard deviation that is close to lum while not producing out-of-range values, i.e. outside of [0, 255].

            Args:
                images (ImageListType): A list of grayscale images to be processed.
                masks (ImageListType): Optional. A list of mask(s) for figure-ground segregation.
                    Each mask contains ones where the histograms are obtained (e.g., foreground) and zeros elsewhere.
                lum (Iterable[Union[float, int]]): Optional. An iterable of the requested mean and standard deviation.
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


        # Validate lum arg
        if lum is not None and not (isinstance(lum, Iterable) and all([isinstance(item, (float, int)) for item in lum]) and len(lum) ==2):
            raise ValueError("lum should be an iterable of two numbers")
        if lum is None:
            lum = (0, 0)
        if not (lum[0] >= 0 and lum[0] <= 255):
            raise ValueError(f"Mean luminance is {lum[0]} but should be between 0 and 255")
        if lum[1] < 0:
            raise ValueError(f"Standard deviation is {lum[1]} but should be greater than or equal to 0")

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
        target_mean, target_std = (np.mean(original_means), np.mean(original_stds)) if lum == (0, 0) or lum is None else lum
        predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
        print(f"Target values: M = {target_mean:.4f}, SD = {target_std:.4f}") if self.verbose else None
        if safe_values and (any(predicted_min<0) or any(predicted_max>255)):
            max_range = predicted_max.max() - predicted_min.min()
            scaling_factor = min(1, (255 - 1e-6) / max_range)  # Safety margin of 1e-6 to avoid precision issues
            target_std *= scaling_factor
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            target_mean = target_mean + (255 - np.max(predicted_max))
            print(f"Adjusted target values for safe values: M = {target_mean:.4f}, SD = {target_std:.4f}") if self.verbose else None
            predicted_min, predicted_max, predicted_range = predict_values(original_means, original_stds, original_min_max, target_mean, target_std)
            if (any(predicted_min < 0) or any(predicted_max > 255)):
                raise Exception(f'Out-of-range values detected: mins = {list(predicted_min)}, maxs = {list(predicted_max)}')

        for idx, im in enumerate(self.dataset.images):
            im2 = im.copy().astype(float)
            M = MatlabOperators.mean2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.mean(im2[self.bool_masks[idx]])
            SD = MatlabOperators.std2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.std(im2[self.bool_masks[idx]])
            print(f'Image #{idx}:\n\tOriginal:\t\tM = {M:.4f}, SD = {SD:.4f}') if self.verbose else None

            # Standardization
            if original_stds[idx] != 0:
                im2[self.bool_masks[idx]] = (im2[self.bool_masks[idx]] - original_means[idx]) / original_stds[idx] * target_std + target_mean
            else:
                im2[self.bool_masks[idx]] = target_mean

            M = MatlabOperators.mean2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.mean(im2[self.bool_masks[idx]])
            SD = MatlabOperators.std2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.std(im2[self.bool_masks[idx]])
            print(f'\tStandardized:\tM = {M:.4f}, SD = {SD:.4f}') if self.verbose else None
            mx, mn = np.max(im2[self.bool_masks[idx]]), np.min(im2[self.bool_masks[idx]])
            clipping_needed = mn<0 or mx>255
            print(f"{bcolors.WARNING}Warning: Clipping applied because values of image #{idx} are outside the [0, 255] range: [{mn}, {mx}]. Results of lum_match might not be exact{bcolors.ENDC}") if self.verbose and clipping_needed else None
            im2 = MatlabOperators.uint8(im2) if self.options.legacy_mode else np.clip(im2, 0, 255).astype('uint8')

            # Save resulting image
            M = MatlabOperators.mean2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.mean(im2[self.bool_masks[idx]])
            SD = MatlabOperators.std2(im2[self.bool_masks[idx]]) if self.options.legacy_mode else np.std(im2[self.bool_masks[idx]])
            print(f'\tFinal:\t\t\tM = {M:.4f}, SD = {SD:.4f}') if self.verbose else None
            print(f'\tTarget values:\tM = {target_mean:.4f}, SD = {target_std:.4f}\n') if self.verbose else None
            self.dataset.images[idx] = im2 #update the dataset
            self.dataset.images.drange = (0, 255)

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

        # Cumulative distribution function : gives the proportion of pixel = or under the vlaue of a bin for each channel
        def _count_cdf(hist_count: np.ndarray, normalized: bool=True) -> np.ndarray:
            cdf = np.cumsum(hist_count, axis=0).astype(np.float64)
            if normalized:
                for channel in range(hist_count.shape[1]):
                    cdf[:, channel] /= cdf[-1, channel]
            return cdf

        def _match_count_cdf(image: np.ndarray, mask: np.ndarray, target_cdf: np.ndarray, noise_level: float, n_bins: int = 256):
            noise = np.random.uniform(-noise_level/2, noise_level/2, size=image.shape)

            # Exact histogram specification requires the addition of noise to convert discrete into continuous pixel values
            noisy_image = image + noise
            source_hist = imhist(noisy_image, mask, n_bins=n_bins)
            source_cdf = _count_cdf(source_hist)
            image = im3D(image)
            new_im = np.zeros(image.shape)
            n_bits = int(np.log2(n_bins))

            if np.issubdtype(image.dtype, np.floating):
                image = image.astype(f'uint{n_bits}')
            elif np.iinfo(image.dtype).bits != n_bits:
                raise TypeError(f"image.dtype is {image.dtype} but n_bins argument = {n_bins}")

            for channel in range(image.shape[-1]):
                # Map source intensities to target intensities
                mapping = np.interp(source_cdf[:, channel], target_cdf[:, channel], np.arange(n_bins))
                mapping = np.clip(mapping, 0, n_bins-1)  # Ensure valid intensity values

                # Data and mask of the channel
                channel_data = image[:, :, channel]
                channel_mask = mask[..., channel] if mask.ndim == 3 else mask
                
                new_channel = channel_data.copy()
                new_channel[channel_mask] = mapping[channel_data[channel_mask]]
                new_im[:, :, channel] = new_channel
            return new_im.astype(image.dtype).squeeze()
        
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
                        print('Why does this happen?')
                
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
            
        if hist_specification:
            target_cdf = _count_cdf(target_hist)
        
        # Match the histogram
        n_iter = self.options.iterations  # Number of iterations for SSIM optimization (default = 10)
        step_size = self.options.step_size  # Step size (default = 67)
        for idx, image in enumerate(buffer_collection):
            if hist_optim:
                image = im3D(image)
                X = image.copy() #TODO: Is the copy really needed?
                M = np.prod(image.shape)/image.shape[2]
                for iter in range(n_iter):
                    if hist_specification:
                        Y = _match_count_cdf(image=X, mask=self.bool_masks[idx], target_cdf=target_cdf, noise_level=noise_level, n_bins=n_bins)
                    else:
                        Y, OA = exact_histogram(image=X, target_hist=target_hist, binary_mask=self.bool_masks[idx])                        
                        print(f'Ordering accuracy per channel = {OA}') if self.verbose else None

                    sens, ssim = ssim_sens(image, Y, n_bins=n_bins)
                    print(f'Mean SSIM = {np.mean(ssim):.4f}') if self.verbose else None
                    ssim_update = sens * step_size * M
                    X = Y + ssim_update # X float64, Y uint8/uint16
                    X = np.clip(X, 0, 2**bit_size - 1).astype(np.uint16) if bit_size == 16 else np.clip(X, 0, 2**bit_size - 1)
                new_image = X.astype(np.uint16) if bit_size == 16 else X.astype(np.uint8)
            else:
                if hist_specification == 1:
                    new_image = _match_count_cdf(image, self.bool_masks[idx], target_cdf, noise_level, n_bins)
                else:
                    new_image, OA = exact_histogram(image=image, target_hist=target_hist, binary_mask=self.bool_masks[idx])
                    print(f'Ordering accuracy per channel = {OA}') if self.verbose else None

            buffer_collection[idx] = new_image
    
        output_collection = self._apply_post_processing(output_name=output_name, output_collection=buffer_collection, dithering=self.options.dithering)
        self._set_relevant_output(output_collection, output_name)

    def fourier_match(self, target_spectrum: Optional[np.ndarray] = None, matching_type: str = 'sf', rescaling_option: Optional[int] = 1) -> List[np.ndarray]:
        """
        Match either the rotational average of the Fourier amplitude or the entire spectrum for a set of images.

        Args:
            matching_type (str): if "sf", it will match the rotational average of the fourier. If "spec" it will
                match the entire spectrum.
            target_spectrum (Optional[np.ndarray]) : Target magnitude spectrum. Same size as the images.
                If None, the target magnitude spectrum is the average spectrum of the all the input images.
                    E.g.,
                        fftim = np.fft.fftshift(np.fft.fft2(im))
                        rho, theta = self.cart2pol(np.real(fftim), np.imag(fftim))
                        tarmag = rho
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

        if matching_type not in ['sf', 'spec']:
            raise ValueError("Matching type must be either 'sf' or 'spec'")

        # Apply the relevant Fourier match
        if matching_type == 'sf':
            buffer_collection = _sf_match(input_collection=input_collection, output_collection=buffer_collection, magnitudes=self.dataset.magnitudes, phases=self.dataset.phases, target_spectrum=target_spectrum)
        elif matching_type == 'spec':
            buffer_collection = _spec_match(output_collection=buffer_collection, phases=self.dataset.phases, target_spectrum=target_spectrum)

        # buffer_collection dtype is np.float64 and drange is approx. [-1.5 to 1.5] before rescaling of any sort       
        if self.options.rescaling:
            buffer_collection = rescale_images(buffer_collection, rescaling_option=self.options.rescaling)
        else :
            for idx in range(len(buffer_collection)):
                 buffer_collection[idx] = (np.clip(buffer_collection[idx], 0, 1) * 255).astype(np.uint8)
        
        buffer_collection = self._apply_post_processing(output_name, buffer_collection, dithering=self.options.dithering)
        self._set_relevant_output(buffer_collection, output_name)

