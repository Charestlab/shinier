"""
Unit tests for shinier.ImageDataset

This suite covers both:
  1. Generic Pydantic / structural validation (schema integrity, field defaults)
  2. Specific functional checks on dataset construction, validation, and lifecycle

Run:
    pytest -v tests/unit_tests/test_ImageDataset_unit.py
"""

import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from pydantic import ValidationError
from shinier import ImageDataset, Options, ImageListIO
from tests import utils as utils_test

pytestmark = pytest.mark.unit_tests


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _make_rgb(h: int = 64, w: int = 64, seed: int = 0, dtype: np.dtype = np.uint8) -> np.ndarray:
    """Create a random RGB image array."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return arr.astype(dtype, copy=False)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def imgs_dir(test_tmpdir: Path):
    """Create INPUT folder with 3 RGB PNG images."""
    d = test_tmpdir / "INPUT"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(3):
        arr = _make_rgb(seed=i)
        p = d / f"im_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return d, paths


@pytest.fixture
def masks_dir(test_tmpdir: Path):
    """Create empty MASKS directory."""
    d = test_tmpdir / "MASKS"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# 1. GENERIC VALIDATION TESTS (Pydantic schema, type integrity, defaults)
# =============================================================================
def test_default_instantiation_with_no_args(test_tmpdir: Path):
    """Default init should succeed when folders exist, even if empty."""
    # Create temporary input/output dirs
    inp = test_tmpdir / "INPUT"
    out = test_tmpdir / "OUTPUT"
    inp.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)

    # Symlink 2–3 small test images from utils_test.IMAGE_PATH
    src_images = utils_test.get_small_imgs_path(utils_test.IMAGE_PATH)
    n_images = np.min([5, len(src_images)])
    for idx in range(n_images):
        target = inp / src_images[idx].name
        target.symlink_to(src_images[idx].resolve())

    opt = Options(input_folder=inp, output_folder=out, images_format="png")
    ds = ImageDataset(options=opt)
    assert ds.options.input_folder == inp
    assert ds.images is not None

def test_invalid_options_type_raises():
    """Passing an invalid type for 'options' should raise a Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        ImageDataset(options="not_an_options_object")


def test_assignment_triggers_validation(imgs_dir, test_tmpdir):
    """Changing 'options' or 'images' after creation should keep model valid."""
    inp, paths = imgs_dir
    out = test_tmpdir / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)
    opt = Options(input_folder=inp, output_folder=out, images_format="png")

    ds = ImageDataset(images=paths, options=opt)
    assert ds.images_name == [p.name for p in ds.images.src_paths if p is not None]

    images = []
    for i in range(10):
        images.append(_make_rgb(seed=i))
    ds = ImageDataset(images=images, options=opt)
    assert ds.images.n_images == 10

    images = ImageListIO(input_data=paths)
    ds = ImageDataset(images=images, options=opt)
    old_names = ds.images_name
    ds.options = opt  # should not invalidate model
    assert ds.images_name == old_names


# =============================================================================
# 2. FUNCTIONAL VALIDATION TESTS (Behavioral + domain logic)
# =============================================================================
def test_init_with_explicit_images_and_options(imgs_dir, test_tmpdir: Path):
    """Explicit image list and options should initialize correctly."""
    inp, paths = imgs_dir
    out = test_tmpdir / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)

    opt = Options(
        input_folder=inp, output_folder=out, images_format="png",
        conserve_memory=True, as_gray=0, whole_image=1, mode=1,
    )
    images = ImageListIO(input_data=paths)
    ds = ImageDataset(images=images, options=opt)

    assert ds.n_images == 3
    assert ds.images_name == [p.name for p in paths]
    assert getattr(ds.images, "_n_channels", None) in (1, 3, None)


def test_init_loads_from_options_when_images_none(imgs_dir, test_tmpdir: Path):
    """Dataset loads from options when 'images' is None."""
    inp, _ = imgs_dir
    out = test_tmpdir / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)

    opt = Options(
        input_folder=inp, output_folder=out, images_format="png",
        conserve_memory=False, as_gray=0, whole_image=1, mode=1,
    )
    ds = ImageDataset(images=None, options=opt)

    assert ds.n_images == 3
    assert all(name.endswith(".png") for name in ds.images_name)


def test_validate_raises_if_single_image(test_tmpdir: Path):
    """Validation fails when only one image is present."""
    inp = test_tmpdir / "INPUT"
    out = test_tmpdir / "OUTPUT"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_make_rgb()).save(inp / "one.png")

    opt = Options(input_folder=inp, output_folder=out, images_format="png", whole_image=1, mode=1)
    with pytest.raises(ValueError):
        ImageDataset(images=None, options=opt)


def test_masks_count_must_be_1_or_equal_to_images(imgs_dir, masks_dir, test_tmpdir: Path):
    """If whole_image=3, masks count must be 1 or match n_images."""
    inp, _ = imgs_dir
    out = test_tmpdir / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)
    # create 2 masks, invalid for 3 images
    for i in range(2):
        Image.fromarray(np.full((8, 10), 255, np.uint8)).save(masks_dir / f"mask_{i}.png")

    opt = Options(
        input_folder=inp, output_folder=out, images_format="png",
        whole_image=3, masks_folder=masks_dir, masks_format="png", mode=1,
    )
    with pytest.raises(ValueError):
        ImageDataset(images=None, options=opt)


def test_masks_shape_must_match_images(imgs_dir, masks_dir, test_tmpdir: Path):
    """Mask spatial dimensions must match first image."""
    inp, _ = imgs_dir
    out = test_tmpdir / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((7, 9), 255, np.uint8)).save(masks_dir / "mask.png")

    opt = Options(
        input_folder=inp, output_folder=out, images_format="png",
        whole_image=3, masks_folder=masks_dir, masks_format="png", mode=1,
    )
    with pytest.raises(ValueError):
        ImageDataset(images=None, options=opt)


def test_mode_ge3_creates_magnitude_phase_placeholders(imgs_dir, test_tmpdir: Path):
    """For mode >= 3, placeholders for magnitudes/phases must exist."""
    inp, _ = imgs_dir
    out = test_tmpdir / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)
    opt = Options(
        input_folder=inp, output_folder=out, images_format="png",
        whole_image=1, mode=3, conserve_memory=True,
    )
    ds = ImageDataset(images=None, options=opt)
    assert hasattr(ds, "magnitudes")
    assert hasattr(ds, "phases")
    assert len(ds.magnitudes) == ds.n_images
    assert len(ds.phases) == ds.n_images


def test_save_images_writes_to_output(imgs_dir, test_tmpdir: Path):
    """save_images() should create files in output folder."""
    inp, _ = imgs_dir
    out = test_tmpdir / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)
    opt = Options(
        input_folder=inp, output_folder=out, images_format="png",
        whole_image=1, mode=1, conserve_memory=True,
    )
    ds = ImageDataset(images=None, options=opt)
    ds.save_images()

    saved = list(out.glob("*.png")) + list(out.glob("*.npy"))
    assert len(saved) >= ds.n_images


def test_close_is_idempotent(imgs_dir, test_tmpdir: Path):
    """close() should safely release resources and allow repeated calls."""
    inp, _ = imgs_dir
    out = test_tmpdir / "OUTPUT"
    out.mkdir(parents=True, exist_ok=True)
    opt = Options(
        input_folder=inp, output_folder=out, images_format="png",
        whole_image=1, mode=1, conserve_memory=True,
    )
    ds = ImageDataset(images=None, options=opt)
    ds.close()
    ds.close()  # idempotent

# import numpy as np
# import pytest
# from pathlib import Path
# from PIL import Image
# from shinier import ImageDataset, Options
#
# pytestmark = pytest.mark.unit_tests
#
#
# def _make_rgb(h: int = 8, w: int = 10, seed: int = 0, dtype: np.dtype = np.uint8) -> np.ndarray:
#     """Create a random RGB image array.
#
#     Args:
#         h: Image height.
#         w: Image width.
#         seed: RNG seed.
#         dtype: Numpy dtype for the output array.
#
#     Returns:
#         Random RGB image of shape (h, w, 3) and given dtype.
#     """
#     rng = np.random.default_rng(seed)
#     arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
#     return arr.astype(dtype, copy=False)
#
#
# @pytest.fixture
# def imgs_dir(test_tmpdir: Path):
#     """Create a small INPUT folder with 3 RGB PNG images.
#
#     Args:
#         test_tmpdir: Function-scoped tmp directory from conftest.
#
#     Returns:
#         Tuple of (input_dir, list_of_image_paths).
#     """
#     d = test_tmpdir / "INPUT"
#     d.mkdir(parents=True, exist_ok=True)
#     paths = []
#     for i in range(3):
#         arr = _make_rgb(seed=i)
#         p = d / f"im_{i}.png"
#         Image.fromarray(arr).save(p)
#         paths.append(p)
#     return d, paths
#
#
# @pytest.fixture
# def masks_dir(test_tmpdir: Path):
#     """Create an empty MASKS directory.
#
#     Args:
#         test_tmpdir: Function-scoped tmp directory.
#
#     Returns:
#         Path to the created MASKS directory.
#     """
#     d = test_tmpdir / "MASKS"
#     d.mkdir(parents=True, exist_ok=True)
#     return d
#
#
# def test_init_with_explicit_images_and_options(imgs_dir, test_tmpdir: Path) -> None:
#     """ImageDataset initializes correctly when given explicit image paths.
#
#     Verifies:
#         * n_images and images_name match provided paths.
#         * Channel detection metadata (if present) is coherent.
#     """
#     inp, paths = imgs_dir
#     out = test_tmpdir / "OUTPUT"
#     out.mkdir(parents=True, exist_ok=True)
#
#     opt = Options(
#         input_folder=inp,
#         output_folder=out,
#         images_format="png",
#         conserve_memory=True,
#         as_gray=0,
#         whole_image=1,
#         mode=1,
#     )
#     ds = ImageDataset(images=paths, options=opt)
#
#     assert ds.n_images == 3
#     assert ds.images_name == [p.name for p in paths]
#     nch = getattr(ds.images, "_n_channels", None)
#     assert nch in (1, 3, None)
#
#
# def test_init_loads_from_options_when_images_none(imgs_dir, test_tmpdir: Path) -> None:
#     """ImageDataset loads images from Options when images=None."""
#     inp, _ = imgs_dir
#     out = test_tmpdir / "OUTPUT"
#     out.mkdir(parents=True, exist_ok=True)
#
#     opt = Options(
#         input_folder=inp,
#         output_folder=out,
#         images_format="png",
#         conserve_memory=False,
#         as_gray=0,
#         whole_image=1,
#         mode=1,
#     )
#     ds = ImageDataset(images=None, options=opt)
#
#     assert ds.n_images == 3
#     assert all(n.endswith(".png") for n in ds.images_name)
#
#
# def test_validate_raises_if_single_image(test_tmpdir: Path) -> None:
#     """Validation should fail when only one image is present (where multiple are required)."""
#     inp = test_tmpdir / "INPUT"
#     out = test_tmpdir / "OUTPUT"
#     inp.mkdir(parents=True, exist_ok=True)
#     out.mkdir(parents=True, exist_ok=True)
#
#     arr = _make_rgb(seed=0)
#     Image.fromarray(arr).save(inp / "one.png")
#
#     opt = Options(
#         input_folder=inp,
#         output_folder=out,
#         images_format="png",
#         whole_image=1,
#         mode=1,
#     )
#     with pytest.raises(ValueError):
#         ImageDataset(images=None, options=opt)
#
#
# def test_masks_count_must_be_1_or_equal_to_images(imgs_dir, masks_dir, test_tmpdir: Path) -> None:
#     """If whole_image=3 (masked), masks count must be either 1 or equal to number of images."""
#     inp, paths = imgs_dir
#     out = test_tmpdir / "OUTPUT"
#     out.mkdir(parents=True, exist_ok=True)
#
#     # Create 2 masks (invalid since images=3)
#     for i in range(2):
#         Image.fromarray(np.full((8, 10), 255, dtype=np.uint8)).save(masks_dir / f"mask_{i}.png")
#
#     opt = Options(
#         input_folder=inp,
#         output_folder=out,
#         images_format="png",
#         whole_image=3,
#         masks_folder=masks_dir,
#         masks_format="png",
#         mode=1,
#     )
#     with pytest.raises(ValueError):
#         ImageDataset(images=None, options=opt)
#
#
# def test_masks_shape_must_match_images(imgs_dir, masks_dir, test_tmpdir: Path) -> None:
#     """If a single mask is provided, its shape must match image shapes."""
#     inp, _ = imgs_dir
#     out = test_tmpdir / "OUTPUT"
#     out.mkdir(parents=True, exist_ok=True)
#
#     # One mask with different shape (invalid)
#     Image.fromarray(np.full((7, 9), 255, dtype=np.uint8)).save(masks_dir / "mask.png")
#
#     opt = Options(
#         input_folder=inp,
#         output_folder=out,
#         images_format="png",
#         whole_image=3,
#         masks_folder=masks_dir,
#         masks_format="png",
#         mode=1,
#     )
#     with pytest.raises(ValueError):
#         ImageDataset(images=None, options=opt)
#
#
# def test_mode_ge3_creates_magnitude_phase_placeholders(imgs_dir, test_tmpdir: Path) -> None:
#     """For mode >= 3, ImageDataset should expose magnitudes/phases placeholders if designed so."""
#     inp, _ = imgs_dir
#     out = test_tmpdir / "OUTPUT"
#     out.mkdir(parents=True, exist_ok=True)
#
#     opt = Options(
#         input_folder=inp,
#         output_folder=out,
#         images_format="png",
#         whole_image=1,
#         mode=3,
#         conserve_memory=True,
#     )
#     ds = ImageDataset(images=None, options=opt)
#
#     # Defensive checks (attributes may be optional; assert coherent if present)
#     assert hasattr(ds, "magnitudes")
#     assert hasattr(ds, "phases")
#     assert len(ds.magnitudes) == ds.n_images
#     assert len(ds.phases) == ds.n_images
#
#
# def test_save_images_writes_to_output(imgs_dir, test_tmpdir: Path) -> None:
#     """save_images should materialize outputs (e.g., PNG/NPY) in the output folder."""
#     inp, _ = imgs_dir
#     out = test_tmpdir / "OUTPUT"
#     out.mkdir(parents=True, exist_ok=True)
#
#     opt = Options(
#         input_folder=inp,
#         output_folder=out,
#         images_format="png",
#         whole_image=1,
#         mode=1,
#         conserve_memory=True,
#     )
#     ds = ImageDataset(images=None, options=opt)
#     ds.save_images()
#
#     saved = list(out.glob("*.png")) + list(out.glob("*.npy"))
#     assert len(saved) >= ds.n_images
#
#
# def test_close_is_idempotent(imgs_dir, test_tmpdir: Path) -> None:
#     """Calling close() should not raise and can be called multiple times."""
#     inp, _ = imgs_dir
#     out = test_tmpdir / "OUTPUT"
#     out.mkdir(parents=True, exist_ok=True)
#
#     opt = Options(
#         input_folder=inp,
#         output_folder=out,
#         images_format="png",
#         whole_image=1,
#         mode=1,
#         conserve_memory=True,
#     )
#     ds = ImageDataset(images=None, options=opt)
#     ds.close()
#     ds.close()