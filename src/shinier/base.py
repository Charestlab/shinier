# External package imports
from pydantic import BeforeValidator, BaseModel
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Annotated
from pathlib import Path
import numpy as np


# -----------------------------------------------
# Type definitions
# -----------------------------------------------
# ImageListType = Union[str, Path, List[Union[str, Path]], List[np.ndarray]]
def image_list_validator(value):
    """Validate image list input"""
    if isinstance(value, (str, Path)):
        return value
    if isinstance(value, list):
        if not value:
            raise ValueError("Image list cannot be empty")
        if all(isinstance(item, (str, Path)) for item in value):
            return value
        if all(isinstance(item, np.ndarray) for item in value):
            return value
        raise ValueError("List must contain all paths or all arrays")
    raise ValueError(f"Invalid type: {type(value).__name__}")


ImageListType = Annotated[
    Union[
        str,
        Path,
        List[Union[str, Path]],
        List[np.ndarray]],
    BeforeValidator(image_list_validator)]


class InformativeBaseModel(BaseModel):
    def model_post_init(self, __context: Any) -> None:
        try:
            self.post_init(__context)
        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} post-init failed: {e}") from e

    def post_init(self, __context: Any) -> None:
        pass  # subclasses override
