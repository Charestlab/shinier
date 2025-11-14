from __future__ import annotations

from typing import List
import numpy as np
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Extension


class build_ext(_build_ext):
    """Custom build_ext that injects NumPy's include directory into extensions."""

    def build_extensions(self) -> None:
        """Add NumPy include dir to all extensions before building."""
        numpy_include: str = np.get_include()

        for ext in self.extensions:
            # Ensure include_dirs is a mutable list
            include_dirs: List[str] = list(getattr(ext, "include_dirs", []) or [])
            if numpy_include not in include_dirs:
                include_dirs.append(numpy_include)
            ext.include_dirs = include_dirs

        super().build_extensions()