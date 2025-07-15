# Global imports
from typing import Optional, List
from datetime import datetime
import numpy as np
from pathlib import Path

# Local imports
from shinier import Options
from shinier.utils import ImageListIO, ImageListType


class ImageDataset:
    """
    Class to load and manage a collection of images and masks, keeping track of their state throughout image processing.

    Args:
        images (ImageListType): List of images. If not provided, images will be loaded from `input_folder` as defined in the Options class.
        masks (ImageListType): List of masks, each specifying the parts of the image that should be taken into account. If not provided, they will be loaded from `masks_folder` as defined in the Options class.
        options (Optional[Options]): Instance of the Options class. If not provided, Options will be instantiated with default values.

    Attributes:
        images (ImageListType): The collection of images.
        masks (ImageListType): The collection of masks.
        n_images (int): Number of images.
        n_masks (int): Number of masks.
        images_name (List[str]): List of image file names.
        masks_name (List[str]): List of mask file names.
        processing_steps (List[str]): List of processing steps applied to the dataset.
        options (Options): Configuration options for the dataset.
    """
    def __init__(
        self,
        images: ImageListType = None,
        masks: ImageListType = None,
        options: Optional[Options] = None
    ):
        self.options = options if options else Options()  # Instantiate with default values if not provided.
        self.processing_steps = [] #TODO: use it for versioning instead of numbers?

        # Load images if not provided
        self.images = ImageListIO(
            input_data = images if images else Path(self.options.input_folder) / f"*.{self.options.images_format}",
            conserve_memory = self.options.conserve_memory,
            as_gray = self.options.as_gray,
            save_dir = self.options.output_folder
        )
        self.n_images = len(self.images)
        self.images_name = [path.name for path in self.images.file_paths]
        self.images._n_channels = self.images[0].shape[-1]

        if self.options.whole_image == 3 and self.options.masks_folder != None and self.options.masks_format != None:
            # Load masks if not provided
            self.masks = ImageListIO(
                input_data = masks if masks else Path(self.options.masks_folder) / f"*.{self.options.masks_format}",
                conserve_memory = self.options.conserve_memory,
                as_gray = self.options.as_gray,
                save_dir = self.options.masks_folder
            )
            self.n_masks = len(self.masks)
            self.masks_name = [path.name for path in self.masks.file_paths]

        # Create placeholders for magnitudes and phases if options.mode in [3, 4, 5, 6, 7, 8]
        self.magnitudes, self.phases, self.buffer = None, None, None


        # Create placeholders for buffer if options.mode include hist_match or fourier_match
        if options.mode >= 2:
            # Create placeholders for buffer
            input_data = [np.zeros(self.images[0].shape, dtype=bool) for idx in range(len(self.images))]
            self.buffer = ImageListIO(
                input_data=input_data,
                conserve_memory=self.options.conserve_memory,
                as_gray=self.options.as_gray,
                save_dir = self.options.output_folder
            )

            # Create placeholders for spectra
            if options.mode >= 3:
                self.magnitudes = ImageListIO(
                    input_data=input_data,
                    conserve_memory=self.options.conserve_memory,
                    as_gray=self.options.as_gray
                )
                self.phases = ImageListIO(
                    input_data=input_data,
                    conserve_memory=self.options.conserve_memory,
                    as_gray=self.options.as_gray
                )

        self._validate_dataset()

    def _validate_dataset(self):
        """
        Perform the following checks on the dataset:
        - Masks and images should have compatible sizes if both are provided
        - At least one image to process
        - Number of masks should be either 1 or equal to the number of images.
        """
        if self.n_images <= 1:
            raise ValueError(f"There are {self.n_images} images stored in the dataset. More than one image should be loaded.")
        if self.options.whole_image == 3 and self.options.masks_folder != None and self.options.masks_format != None:
            if self.n_masks > 0:
                if self.n_masks not in [1, self.n_images]:
                    raise ValueError("The number of masks should be either 1 or equal to the number of images.")

            # Ensure masks and images have compatible sizes if both are provided
            if self.masks and self.images:
                if self.masks[0].shape[:2] != self.images[0].shape[:2]:
                    raise ValueError(f"Masks and images should have the same shape")

    def save_images(self):
        self.images.final_save_all()

    def print_log(self) -> None:
        """
        Record processing_steps list for reproducibility
        """

        # Generate a filename with the full date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = Path(self.options.output_folder) / f"log_{current_datetime}.txt"

        # Write each step to a new line in the file
        with open(filename, 'w') as file:
            for step in self.processing_steps:
                file.write(step + '\n')

    def close(self):
        self.images.close()
        if self.options.whole_image == 3 and self.options.masks_folder != None and self.options.masks_format != None:
            self.masks.close()
        if self.magnitudes is not None:
            self.magnitudes.close()
            self.phases.close()

