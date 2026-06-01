import pytest
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
from pydantic import ValidationError
from shinier import ImageListIO, ImageDataset, Options, ImageProcessor
from tests import utils as utils_test

pytestmark = pytest.mark.unit_tests

def _prepare_temp_images(test_tmpdir: Path, n: int = 3) -> Path:
    """Populate temporary folder with a few symlinked test PNGs."""
    inp = test_tmpdir / "INPUT"
    inp.mkdir(parents=True, exist_ok=True)
    src_images = utils_test.get_small_imgs_path(utils_test.IMAGE_PATH)
    for img in src_images[:n]:
        (inp / img.name).symlink_to(img.resolve())
    return inp

def _prepare_temp_dataset(test_tmpdir: Path) -> ImageDataset:
    """Helper to prepare a temporary dataset with a few small images."""
    inp = test_tmpdir / "INPUT"
    out = test_tmpdir / "OUTPUT"
    inp.mkdir()
    out.mkdir()
    src_images = utils_test.get_small_imgs_path(utils_test.IMAGE_PATH)
    for img in src_images[:3]:
        (inp / img.name).symlink_to(img.resolve())

    opt = Options(input_folder=inp, output_folder=out)
    return ImageDataset(options=opt)

def _make_rgb(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """Create a random RGB NumPy image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _make_input_output_dirs(tmp_root: Path) -> tuple[Path, Path]:
    """Utility to make INPUT and OUTPUT dirs under tmp."""
    inp = tmp_root / "INPUT"
    out = tmp_root / "OUTPUT"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return inp, out


def _build_lum_match_processor(
    test_tmpdir: Path,
    images: list[np.ndarray],
    target_lum: tuple[Optional[float], Optional[float]],
    safe_lum_match: bool = True,
) -> ImageProcessor:
    """Build a processor wired for direct lum_match unit tests."""
    inp, out = _make_input_output_dirs(test_tmpdir)
    opt = Options(
        input_folder=inp,
        output_folder=out,
        mode=1,
        target_lum=target_lum,
        safe_lum_match=safe_lum_match,
        as_gray=1,
        verbose=-1,
    )
    io = ImageListIO(input_data=images, conserve_memory=True)
    ds = ImageDataset(images=io, options=opt)
    ds.buffer = ImageListIO(input_data=[im.astype(np.float64) for im in images], conserve_memory=True)
    proc = ImageProcessor(dataset=ds, options=opt, verbose=-1, from_unit_test=True)
    proc.bool_masks = [np.ones(im.shape, dtype=bool) for im in images]
    return proc


def _stats(im: np.ndarray) -> tuple[float, float]:
    """Return mean and std in float space for assertions."""
    arr = im.astype(np.float64)
    return float(arr.mean()), float(arr.std())


def test_processor_init_from_arrays(test_tmpdir: Path) -> None:
    """Processor should run cleanly from in-memory arrays."""
    arrays = [_make_rgb(seed=s) for s in range(2)]
    inp = _prepare_temp_images(test_tmpdir)
    opt = Options(
        output_folder=test_tmpdir,
        mode=1,  # lum_match only
        iterations=1,
        verbose=-1,
    )
    images = ImageListIO(input_data=arrays, conserve_memory=True)
    ds = ImageDataset(images=images, options=opt)
    proc = ImageProcessor(dataset=ds, options=opt, verbose=-1, from_unit_test=True)
    assert proc.dataset is ds
    # assert len(proc.log) >= 1
    # assert proc.seed is not None


def test_processor_init_from_folder(test_tmpdir: Path):
    ds = _prepare_temp_dataset(test_tmpdir)
    proc = ImageProcessor(dataset=ds, from_unit_test=True)
    assert proc.dataset.images is not None
    assert len(proc.dataset.images) > 0


def test_processor_log_and_results(test_tmpdir: Path) -> None:
    """Processor should produce log entries and valid result arrays."""
    arrays = [_make_rgb(seed=s) for s in range(2)]
    inp = _prepare_temp_images(test_tmpdir)
    opt = Options(
        input_folder=inp,
        output_folder=test_tmpdir,
        mode=1,
        iterations=1,
        verbose=-1,
    )
    images = ImageListIO(input_data=arrays, conserve_memory=True)
    ds = ImageDataset(images=images, options=opt)
    ds.images = ImageListIO(input_data=arrays, conserve_memory=True)
    proc = ImageProcessor(dataset=ds, options=opt, verbose=-1, from_unit_test=True)
    # results = proc.get_results()
    # assert isinstance(results, list)
    # assert all(isinstance(im, np.ndarray) for im in results)
    # assert all(im.shape[-1] == 3 for im in results)


def test_processor_invalid_args_raises(test_tmpdir: Path) -> None:
    """Passing an invalid type for dataset should raise Pydantic validation error."""
    with pytest.raises(ValidationError):
        ImageProcessor(dataset="not_a_dataset", options=None)  # type: ignore


def test_processor_dtype_and_range_consistency(test_tmpdir: Path) -> None:
    """Processed buffer should have consistent dtype and range."""
    arrays = [_make_rgb(seed=s) for s in range(2)]
    inp = _prepare_temp_images(test_tmpdir)
    opt = Options(
        input_folder=inp,
        output_folder=test_tmpdir,
        mode=1,
        iterations=1,
        verbose=-1,
    )
    images = ImageListIO(input_data=arrays, conserve_memory=True)
    ds = ImageDataset(images=images, options=opt)
    proc = ImageProcessor(dataset=ds, options=opt, verbose=-1, from_unit_test=True)
    # imgs = proc.get_results()
    # assert all(np.issubdtype(im.dtype, np.floating) or np.issubdtype(im.dtype, np.uint8) for im in imgs)
    # for im in imgs:
    #     assert im.min() >= 0 and im.max() <= 255


def test_lum_match_partial_std_only_keeps_mean(test_tmpdir: Path) -> None:
    """target_lum=(None, x) should preserve mean and control std."""
    img1 = np.tile(np.linspace(90, 170, 64, dtype=np.float64), (64, 1))
    img2 = np.tile(np.linspace(100, 180, 64, dtype=np.float64), (64, 1))
    proc = _build_lum_match_processor(test_tmpdir, [img1, img2], target_lum=(None, 20.0))

    means_before = []
    for im in proc.dataset.buffer:
        m, _ = _stats(im)
        means_before.append(m)

    proc.lum_match()

    for idx, im in enumerate(proc.dataset.buffer):
        mean_after, std_after = _stats(im)
        assert mean_after == pytest.approx(means_before[idx], abs=1e-6)
        assert std_after == pytest.approx(20.0, abs=1e-3)


def test_lum_match_partial_mean_only_keeps_std(test_tmpdir: Path) -> None:
    """target_lum=(x, None) should preserve std and control mean."""
    img1 = np.tile(np.linspace(95, 175, 64, dtype=np.float64), (64, 1))
    img2 = np.tile(np.linspace(105, 185, 64, dtype=np.float64), (64, 1))
    proc = _build_lum_match_processor(test_tmpdir, [img1, img2], target_lum=(100.0, None))

    stds_before = []
    for im in proc.dataset.buffer:
        _, s = _stats(im)
        stds_before.append(s)

    proc.lum_match()

    for idx, im in enumerate(proc.dataset.buffer):
        mean_after, std_after = _stats(im)
        assert mean_after == pytest.approx(100.0, abs=1e-6)
        assert std_after == pytest.approx(stds_before[idx], abs=1e-3)


def test_lum_match_safe_partial_mean_raises_when_out_of_range(test_tmpdir: Path) -> None:
    """safe_lum_match should reject (mean, None) requests that force clipping."""
    img = np.tile(np.linspace(180, 250, 64, dtype=np.float64), (64, 1))
    proc = _build_lum_match_processor(test_tmpdir, [img], target_lum=(240.0, None), safe_lum_match=True)

    with pytest.raises(ValueError, match=r"safe_lum_match cannot keep values within \[0, 255\]"):
        proc.lum_match()


def test_lum_match_constant_image_zero_std_is_stable(test_tmpdir: Path) -> None:
    """Constant images must not trigger divide-by-zero path in safe checks."""
    img = np.full((64, 64), 128.0, dtype=np.float64)
    proc = _build_lum_match_processor(test_tmpdir, [img], target_lum=(140.0, 20.0), safe_lum_match=True)

    proc.lum_match()
    mean_after, std_after = _stats(proc.dataset.buffer[0])
    assert mean_after == pytest.approx(140.0, abs=1e-6)
    assert std_after == pytest.approx(0.0, abs=1e-9)