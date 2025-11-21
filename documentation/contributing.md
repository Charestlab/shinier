```text
   ███████╗██╗  ██╗██╗███╗  ██╗██╗███████╗██████╗
   ██╔════╝██║  ██║██║████╗ ██║██║██╔════╝██╔══██╗
   ███████╗███████║██║██╔██╗██║██║█████╗  ██████╔╝
   ╚════██║██╔══██║██║██║╚████║██║██╔══╝  ██╔══██╗
   ███████║██║  ██║██║██║ ╚███║██║███████╗██║  ██║
   ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚══╝╚═╝╚══════╝╚═╝  ╚═╝
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![PyPI version](https://img.shields.io/pypi/v/shinier.svg)](https://pypi.org/project/shinier/)
---

# Contributing to SHINIER

Thanks for your interest in improving **SHINIER**. This document explains how to set up a development environment, coding standards, testing strategy (unit vs. validation), and the pull-request process. Please read it fully before opening a PR.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Development Setup](#development-setup)
3. [Branching & Commits](#branching--commits)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Performance & Memory Guidelines](#performance--memory-guidelines)
7. [Documentation](#documentation)
8. [PR Checklist](#pr-checklist)
9. [Issue Triage & Feature Proposals](#issue-triage--feature-proposals)
10. [Release Hygiene (maintainers)](#release-hygiene-maintainers)

---

## Code of Conduct
By participating, you agree to uphold a standard of professional, inclusive, and respectful collaboration. If your organization already uses a Code of Conduct, link it here (e.g., Contributor Covenant). Reports can be sent to the maintainers.

---

## Development Setup

> **Python:** >=3.9, <3.13  
> **OS:** macOS / Linux / Windows  
> **Optional:** C/C++ toolchain for the Cython-compiled `_cconvolve` extension (speeds up convolution)

```bash
# 1) Fork on GitHub, then clone your fork
git clone https://github.com/Charestlab/shinier.git
cd shinier

# 2) Create a virtual environment (example: venv)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3) Editable install with dev tools
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"              # includes pytest and external packages for validation

# 4) Optional: ensure a compiler is available (for fast Cython extension)
# macOS:   xcode-select --install
# Ubuntu:  sudo apt update && sudo apt install -y build-essential python3-dev
# Windows: Install "Build Tools for Visual Studio" (C++)

# 5) Pre-commit hooks (formatting/linting on commit)
pre-commit install

# 6) Quick smoke test
pytest -m unit_tests
```

## Coding Standards
### Git rules
- No one has push permissions on `main`. 
- PR requires 2 approvals + all tests to pass before merging. 
- Official working developpement branch is dev_X, where X is the subversion of the last Pypi release (e.g. if release is 0.1.7, then branch is dev_1.7). 

### To do before a PR
- Merge main on dev_X branch (not the other way around).
- Make sure all status checks are green.
- Initiate PR

### Language & typing
- Use **type hints everywhere** (functions, methods, Pydantic models, tests).
- Prefer explicit array types (e.g., `np.ndarray`) and document expected **shape** and **dtype**.
- Keep functions small and pure when possible; avoid implicit mutation unless documented.

### Docstrings
- Use **Google-style** docstrings for all public functions/classes/methods.

```python
import numpy as np

def lum_match(img: np.ndarray, target: float) -> np.ndarray:
    """Match the mean luminance of an image to a target.

    Args:
        img: Image array in float space, range [0, 1], shape (H, W) or (H, W, 3).
        target: Target mean luminance in [0, 1].

    Returns:
        A new image array with adjusted mean luminance (same shape as input).

    Raises:
        TypeError: If ``img`` is not a NumPy array.
        ValueError: If ``target`` is outside [0, 1] or ``img`` range is invalid.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy.ndarray.")
    if not (0.0 <= float(target) <= 1.0):
        raise ValueError("target must be in [0, 1].")
    if img.size == 0:
        return img.copy()

    current = float(np.clip(img, 0.0, 1.0).mean())
    if current == 0.0:
        # Avoid division by zero; return image filled with target
        return np.full_like(img, target, dtype=img.dtype)

    scale = target / current
    out = np.clip(img.astype(np.float64) * scale, 0.0, 1.0)
    return out.astype(img.dtype, copy=False)
```


### Pydantic models
- Core classes should be **Pydantic v2** models inheriting our customized base-model: `InformativeBaseModel`
- Note that `InformativeBaseModel` uses a `post_init` method that wraps pydantic's `model_post_init`
- Use `Field` for defaults/descriptions; prefer **strict** types when coercion is dangerous.
- Validate with `@field_validator` / `@model_validator` and raise **specific** exceptions.
- Avoid hidden coercions (e.g., `2 -> True`). Use `StrictBool` for boolean flags.
- Keep `.model_dump()` and `.model_json_schema()` coherent and documented.

```python
from typing import Literal, Optional, Any
import numpy as np
from shinier.base import InformativeBaseModel
from shinier import REPO_ROOT
from pydantic import Field, ConfigDict, StrictBool, field_validator, model_validator
from pathlib import Path


class Options(InformativeBaseModel):
    """..."""

    model_config = ConfigDict(
        validate_assignment=True,  # Validate every time object updated
        extra="forbid",  # Does not allow unknown attributes
        arbitrary_types_allowed=True,  # Allow non-pydantic types (e.g. np.ndarray)
    )

    # --- I/O ---
    input_folder: Optional[Path] = Field(default=REPO_ROOT / "INPUT")
    output_folder: Path = Field(default=REPO_ROOT / "OUTPUT")

    # --- Masks ---
    masks_folder: Optional[Path] = Field(default=None)

    """..."""

    @field_validator("input_folder", "output_folder", "masks_folder")
    @classmethod
    def validate_existing_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None:
            v = v.resolve()
            if not v.exists():
                raise ValueError(f"Folder does not exist: {v}")
        return v

    """..."""
    def post_init(self, __context: Any) -> None:
        """Put your initialization logic here. It will run after Pydantic validation and only once at instantiation."""
        
```

### Exceptions
- **Never** use `bare except:` or `except Exception:`. Catch **specific** exceptions.
- Error messages must be actionable (what failed + how to fix).

```python
import numpy as np

def safe_std(a: np.ndarray) -> float:
    """Return standard deviation with clear failure modes."""
    if not isinstance(a, np.ndarray):
        raise TypeError("a must be a numpy.ndarray.")
    try:
        return float(np.std(a, dtype=np.float64))
    except (FloatingPointError, ValueError) as err:
        raise ValueError(f"std failed: {err}") from err
```

### Unit-Tests and Validation Tests
- Run locally and keep CI green.
- Recommended tools: **pytest**.

```bash

# Tests (fast) and markers
pytest -m unit_tests

# Validation Tests (VERY LONG AND SLOW) and markers
pytest -q -m validation_tests
```
#### 🏛️ Test README

