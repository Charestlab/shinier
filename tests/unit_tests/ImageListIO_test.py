import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from pydantic import ValidationError
from shinier import ImageListIO

pytestmark = pytest.mark.unit_tests


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _make_rgb(h: int = 8, w: int = 10, seed: int = 0) -> np.ndarray:
    """Create a reproducible RGB NumPy array."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------
# Core tests — initialization and cleanup
# ---------------------------------------------------------------------
def test_init_from_arrays_creates_temp_and_cleans_on_close(test_tmpdir: Path) -> None:
    """ImageListIO created from arrays should allocate a temp root that is cleaned on close."""
    arrays = [_make_rgb(seed=s) for s in range(2)]
    coll = ImageListIO(input_data=arrays, conserve_memory=True)
    tmp_dir = getattr(coll, "_temp_dir", None)

    assert tmp_dir is not None and tmp_dir.exists()
    coll.close()
    after = getattr(coll, "_temp_dir", None)
    assert after is None or not tmp_dir.exists()


def test_iter_and_len_from_arrays(test_tmpdir: Path) -> None:
    """Length and iteration should work for in-memory arrays."""
    arrays = [_make_rgb(seed=s) for s in range(3)]
    coll = ImageListIO(input_data=arrays, conserve_memory=True)

    assert len(coll) == 3
    assert sum(1 for _ in coll) == 3
    coll.close()


def test_init_from_folder_detects_images(test_tmpdir: Path) -> None:
    """When initialized from a folder wildcard, collection should discover and load images."""
    inp = test_tmpdir / "INPUT"
    inp.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        Image.fromarray(_make_rgb(seed=i)).save(inp / f"im_{i}.png")

    coll = ImageListIO(input_data=str(inp / "*.png"), conserve_memory=True)
    assert len(coll) == 2
    assert coll.reference_size == coll.data[0].shape[:2]
    coll.close()


# ---------------------------------------------------------------------
# Gray mode and dtype handling
# ---------------------------------------------------------------------
def test_as_gray_flag_validation() -> None:
    """Invalid as_gray values should trigger ValueError at post-init."""
    arrays = [_make_rgb()]
    with pytest.raises(ValueError):
        ImageListIO(input_data=arrays, as_gray=9)  # invalid literal


def test_as_gray_flag_accepted(test_tmpdir: Path) -> None:
    """as_gray flag should be correctly forwarded and retained."""
    arrays = [_make_rgb(seed=s) for s in range(2)]
    coll = ImageListIO(input_data=arrays, conserve_memory=True, as_gray=3)
    assert coll.as_gray == 3
    coll.close()


def test_dtype_and_range_consistency(test_tmpdir: Path) -> None:
    """All images should share dtype and dynamic range."""
    arrays = [_make_rgb(seed=s) for s in range(2)]
    coll = ImageListIO(input_data=arrays, conserve_memory=False)
    assert coll.dtype == np.uint8
    assert coll.drange == (0, 255)
    coll.close()


# ---------------------------------------------------------------------
# Pydantic behavior and schema tests
# ---------------------------------------------------------------------
def test_pydantic_validation_errors() -> None:
    """Invalid input types should raise ValidationError."""
    with pytest.raises(ValidationError):
        ImageListIO(input_data=123)  # invalid type

    with pytest.raises(ValidationError):
        ImageListIO(input_data={"a": "b"})  # wrong type entirely


def test_model_instantiation_defaults(test_tmpdir: Path) -> None:
    """Minimal valid construction should succeed."""
    arrays = [_make_rgb()]
    model = ImageListIO(input_data=arrays)
    assert model.n_images >= 1
    assert model.save_dir.exists()
    model.close()


def test_pydantic_serialization_roundtrip(test_tmpdir: Path) -> None:
    """Model should serialize and re-instantiate cleanly."""
    arrays = [_make_rgb(seed=1)]
    model = ImageListIO(input_data=arrays)
    data = model.model_dump()
    assert isinstance(data, dict)
    restored = ImageListIO(**data)
    assert isinstance(restored, ImageListIO)
    model.close()
    restored.close()


# ---------------------------------------------------------------------
# Filesystem and read-only behavior
# ---------------------------------------------------------------------
def test_final_save_and_reopen(tmp_path: Path) -> None:
    """Saving to disk should persist images to a known directory."""
    out = tmp_path / "OUT"
    arrays = [_make_rgb(seed=s) for s in range(2)]
    coll = ImageListIO(input_data=arrays, conserve_memory=True, save_dir=out)
    coll.final_save_all()
    saved = list(out.glob("*"))
    assert len(saved) >= 2
    coll.close()


def test_readonly_copy_behaves_identically(test_tmpdir: Path) -> None:
    """readonly_copy should produce a mirror with same metadata and read-only flag."""
    arrays = [_make_rgb(seed=s) for s in range(2)]
    coll = ImageListIO(input_data=arrays, conserve_memory=True)
    clone = coll.readonly_copy()
    assert clone.n_images == coll.n_images
    assert clone._read_only is True
    with pytest.raises(RuntimeError):
        clone[0] = _make_rgb()  # should not allow modification
    coll.close()
    clone.close()