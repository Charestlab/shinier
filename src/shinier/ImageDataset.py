from __future__ import annotations
from typing import Optional, List, Literal, Union, Any
import numpy as np
from pathlib import Path
from pydantic import Field, ConfigDict

# Local imports
from shinier import Options
from .ImageListIO import ImageListIO
from shinier.utils import print_log
from shinier.base import ImageListType, InformativeBaseModel


class ImageDataset(InformativeBaseModel):
    """
    Class to load and manage a collection of images and masks, keeping track of their state throughout image processing.

    Args:
        images (ImageListType): List of images. If not provided, images will be loaded from `input_folder` as defined in the Options class.
        masks (ImageListType): List of masks, each specifying the parts of the image that should be taken into account.
            If not provided, they will be loaded from `masks_folder` as defined in the Options class.
        options (Optional[Options]): Instance of the Options class. If not provided, Options will be instantiated with default values.

    Attributes:
        images (ImageListType): The collection of images.
        masks (ImageListType): The collection of masks.
        n_images (int): Number of images.
        n_masks (int): Number of masks.
        images_name (List[str]): List of image file names.
        masks_name (List[str]): List of mask file names.
        processing_logs (List[str]): List of processing steps applied to the dataset along with relevant image metrics and information.
        options (Options): Configuration options for the dataset.
    """

    # --- Pydantic config ---
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # allow np.ndarray, Path, ImageListIO, etc.
        extra="forbid",
        validate_assignment=True,
    )

    # --- User-provided / externally settable attributes ---
    images: Optional[Union[ImageListIO, ImageListType]] = None
    masks: Optional[Union[ImageListIO, ImageListType]] = None
    options: Options = Field(default_factory=Options)

    # --- Internally constructed / derived attributes ---
    processing_logs: List[str] = Field(default_factory=list)
    n_images: Optional[int] = None
    n_masks: Optional[int] = None
    images_name: Optional[List[str]] = None
    masks_name: Optional[List[str]] = None

    magnitudes: Optional[ImageListIO] = None
    phases: Optional[ImageListIO] = None
    buffer: Optional[ImageListIO] = None
    buffer_other: Optional[ImageListIO] = None

    # ----------------------------------------------------------------------
    # Post-init hook — runs after Pydantic validation, but before returning
    # ----------------------------------------------------------------------
    def post_init(self, __context: Any) -> None:
        """Called automatically after Pydantic validation."""

        # Initialization of attributes
        self.initialize_dataset()

        # Internal validation
        self._validate_dataset()

    # =====================================================================================
    # Post-initialization logic (dataset loading, validation, buffer setup)
    # =====================================================================================

    def initialize_dataset(self) -> ImageDataset:
        """Perform dataset construction and validation after field initialization."""

        # Determine grayscale behavior
        as_gray: Literal[0, 1, 2, 3, 4] = 1 if self.options.as_gray and self.options.linear_luminance else 0

        # ------------------------- Load images -------------------------
        if isinstance(self.images, ImageListIO):
            # Make new shallow copy and force run initialization script (model_post_init)
            # to account for updated values (e.g. as_gray)
            images = self.images.model_copy(update={"as_gray": as_gray, "conserve_memory": self.options.conserve_memory, "save_dir": self.options.output_folder})
            images.model_post_init(None)
        else:
            images = ImageListIO(
                input_data=self.images if self.images else Path(self.options.input_folder) / "*",
                conserve_memory=self.options.conserve_memory,
                as_gray=as_gray,  # Now done in ImageProcessor (previous code: self.options.as_gray)
                save_dir=self.options.output_folder
            )
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "n_images", len(images))
        object.__setattr__(self, "images_name", [p.name for p in images.src_paths if p is not None])

        # ------------------------- Load masks if required -------------------------
        if self.options.whole_image == 3 and self.options.masks_folder:
            if isinstance(self.masks, ImageListIO):
                # Make new shallow copy and force run initialization script (model_post_init)
                # to account for updated values (e.g. as_gray)
                masks = self.masks.model_copy(
                    update={"as_gray": as_gray, "conserve_memory": self.options.conserve_memory,
                            "save_dir": self.options.masks_folder})
                masks.model_post_init(None)
            else:
                masks = ImageListIO(
                    input_data=self.masks if self.masks else Path(self.options.masks_folder) / f"*",
                    conserve_memory=self.options.conserve_memory,
                    as_gray=as_gray,  # Now done in ImageProcessor (previous code: self.options.as_gray)
                    save_dir=self.options.masks_folder,
                )
            object.__setattr__(self, "masks", masks)
            object.__setattr__(self, "n_masks", len(self.masks))
            object.__setattr__(self, "masks_name", [p.name for p in self.masks.src_paths])

        # ------------------------- Allocate buffers -------------------------
        # Create placeholders for magnitudes and phases if options.mode in [3, 4, 5, 6, 7, 8]
        self.magnitudes, self.phases, self.buffer = None, None, None
        buffer_size = self.images[0].shape[:2] if not self.options.linear_luminance or self.options.as_gray == 1 else self.images[0].shape

        # Create placeholders for buffers
        input_data = [np.zeros(buffer_size, dtype=bool) for idx in range(len(self.images))]
        buffer = ImageListIO(
            input_data=input_data,
            conserve_memory=True,  # <--- FORCE
            # conserve_memory=self.options.conserve_memory,
            as_gray=0,  # Now done in ImageProcessor (previous code: self.options.as_gray)
            save_dir=self.options.output_folder
        )
        object.__setattr__(self, "buffer", buffer)

        # Create placeholders for buffer_ab if as_gray==0 and linear_luminance is False
        if not self.options.linear_luminance:
            buffer_other = ImageListIO(
                input_data=input_data,
                conserve_memory=True,  # <--- FORCE
                # conserve_memory=self.options.conserve_memory,
                as_gray=0,  # Now done in ImageProcessor (previous code: self.options.as_gray)
                save_dir=self.options.output_folder
            )
            object.__setattr__(self, "buffer_other", buffer_other)

        # Create placeholders for spectra
        if self.options.mode >= 3 and self.options.mode != 9:
            magnitudes = ImageListIO(
                input_data=input_data,
                conserve_memory=True,  # <--- FORCE
                # conserve_memory=self.options.conserve_memory,
                as_gray=0,  # Now done in ImageProcessor (previous code: self.options.as_gray)
            )
            object.__setattr__(self, "magnitudes", magnitudes)
            phases = ImageListIO(
                input_data=input_data,
                conserve_memory=True,  # <--- FORCE
                # conserve_memory=self.options.conserve_memory,
                as_gray=0,  # Now done in ImageProcessor (previous code: self.options.as_gray)
            )
            object.__setattr__(self, "phases", phases)

        return self

    # =====================================================================================
    # Internal validation
    # =====================================================================================
    def _validate_dataset(self) -> None:
        """
        Perform the following checks on the dataset:
        - Masks and images should have compatible sizes if both are provided
        - At least one image to process
        - Number of masks should be either 1 or equal to the number of images.
        """
        if self.n_images is None or self.n_images < 1:
            raise ValueError(f"Invalid dataset: {self.n_images} images found. At least one images is required.")

        if self.options.whole_image == 3 and self.options.masks_folder:
            if self.n_masks not in (1, self.n_images):
                raise ValueError("Number of masks must be 1 or match number of images.")

            if self.masks and self.images:
                if self.masks[0].shape[:2] != self.images[0].shape[:2]:
                    raise ValueError("Masks and images must have identical spatial dimensions.")

    # =====================================================================================
    # Lifecycle helpers
    # =====================================================================================
    def save_images(self) -> None:
        """Save processed images to disk."""
        self.images.final_save_all()

    def print_log(self) -> None:
        """Record processing steps to a timestamped log file."""
        print_log(logs=self.processing_logs, log_path=Path(self.options.output_folder))

    def close(self) -> None:
        """Release all loaded image and buffer resources."""
        if self.images:
            self.images.close()
        if self.masks:
            self.masks.close()
        for attr in ("magnitudes", "phases", "buffer", "buffer_other"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.close()

    def __setattr__(self, name, value):
        """Prevent reassignment of `images` if dimensions/metadata differ."""
        if name == "images" and hasattr(self, "images") and getattr(self, "images") is not None:
            # Only enforce the check if the new value is an ImageListIO
            old = getattr(self, "images")
            if isinstance(value, type(old)):
                for field in ["reference_size", "n_dims", "n_images", "n_channels"]:
                    if not (hasattr(value, field) and hasattr(old, field) and getattr(value, field) == getattr(old, field)):
                        raise ValueError(
                            "Cannot reassign `images`: new dataset does not match "
                            "reference size, n_dims, n_images, or n_channels."
                        )
        super().__setattr__(name, value)
