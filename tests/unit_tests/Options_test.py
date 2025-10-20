import numpy as np
from pathlib import Path
from shinier import Options
import pytest

pytestmark = pytest.mark.unit_tests


@pytest.fixture
def tmp_io_dirs(tmp_path):
    inp = tmp_path / "INPUT"
    out = tmp_path / "OUTPUT"
    inp.mkdir()
    out.mkdir()
    return inp, out


def make(inp, out, **kwargs):
    """Helper to construct Options with required input/output dirs."""
    return Options(input_folder=inp, output_folder=out, **kwargs)


def test_legacy_mode_skips_validation_and_sets_conserve_memory_false(tmp_path):
    # No folders needed because legacy_mode skips validation
    opt = Options(legacy_mode=True)
    assert opt.conserve_memory is False


def test_nonexistent_input_or_output_raises(tmp_path):
    missing = tmp_path / "MISSING"
    with pytest.raises(ValueError):
        Options(input_folder=missing, output_folder=missing)


def test_happy_path_minimal(tmp_io_dirs):
    inp, out = tmp_io_dirs
    opt = Options(input_folder=inp, output_folder=out)
    # Basic coercions / defaults
    assert isinstance(opt.input_folder, Path)
    assert isinstance(opt.output_folder, Path)
    assert opt.images_format in {"png", "tif", "tiff", "jpg", "jpeg"}
    assert opt.mode == 8
    assert opt.as_gray == 0
    assert opt.dithering == 1
    assert opt.conserve_memory is True


@pytest.mark.parametrize("fmt", ["png", "tif", "tiff", "jpg", "jpeg"])
def test_valid_image_formats(tmp_io_dirs, fmt):
    inp, out = tmp_io_dirs
    opt = Options(input_folder=inp, output_folder=out, images_format=fmt)
    assert opt.images_format == fmt


def test_invalid_images_format_raises(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(ValueError, match="images format must be either"):
        make(inp, out, images_format="bmp")


def test_masks_require_valid_format_when_using_masks(tmp_io_dirs, tmp_path):
    inp, out = tmp_io_dirs
    mask_dir = tmp_path / "MASK"
    mask_dir.mkdir()
    # OK when using masks with proper format
    ok = make(inp, out, whole_image=3, masks_folder=mask_dir, masks_format="png")
    assert ok.masks_folder == mask_dir
    # Invalid format
    with pytest.raises(ValueError, match="masks format must be either"):
        make(inp, out, whole_image=3, masks_folder=mask_dir, masks_format="gif")


def test_missing_masks_folder_raises_when_using_masks(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(TypeError):
        make(inp, out, whole_image=3)  # no masks_folder passed


@pytest.mark.parametrize("val", [0, 4, -1, 99])
def test_whole_image_invalid_values(tmp_io_dirs, val):
    inp, out = tmp_io_dirs
    with pytest.raises(ValueError, match="whole_image must be 1, 2 or 3"):
        make(inp, out, whole_image=val)


@pytest.mark.parametrize("val", [1, 9])
def test_valid_modes(tmp_io_dirs, val):
    inp, out = tmp_io_dirs
    opt = make(inp, out, mode=val)
    assert opt.mode == val


def test_invalid_mode_raises(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(ValueError, match="Invalid mode selected"):
        make(inp, out, mode=10)


@pytest.mark.parametrize("val", [0, 1, 2])
def test_as_gray_valid(tmp_io_dirs, val):
    inp, out = tmp_io_dirs
    opt = make(inp, out, as_gray=val)
    assert opt.as_gray == val


def test_as_gray_invalid(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(TypeError, match="as_gray must be an int equal to 0, 1, 2, 3 or 4."):
        make(inp, out, as_gray=5)


@pytest.mark.parametrize("val", [0, 1, 2])
def test_dithering_valid(tmp_io_dirs, val):
    inp, out = tmp_io_dirs
    opt = make(inp, out, dithering=val)
    assert opt.dithering == val


def test_dithering_invalid(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(ValueError, match="dithering must be an int equal to 0, 1 or 2"):
        make(inp, out, dithering=4)


def test_seed_invalid_type(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(TypeError, match="seed must be an integer value or None"):
        make(inp, out, seed="abc")


def test_target_lum_validation(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(ValueError, match="target_lum should be an iterable of two numbers"):
        make(inp, out, target_lum=(1,))
    # Valid edge values
    ok = make(inp, out, target_lum=(0, 0))
    assert ok.target_lum == (0, 0)


def test_hist_specification_and_optim(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(ValueError, match="hist_specification must be 1, 2, 3 or 4"):
        make(inp, out, hist_specification=5)
    with pytest.raises(ValueError, match="Optim must be 0 or 1"):
        make(inp, out, hist_optim=3)


def test_hist_iterations_and_step_size(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(ValueError, match="hist_iterations must be at least 1"):
        make(inp, out, hist_iterations=0)
    with pytest.raises(ValueError, match="Step size must be at least 1"):
        make(inp, out, step_size=0)


def test_rescaling_values(tmp_io_dirs):
    inp, out = tmp_io_dirs
    for val in (0, 1, 2, 3):
        assert make(inp, out, rescaling=val).rescaling == val
    with pytest.raises(ValueError, match="Rescaling must be 0, 1, 2 or 3"):
        make(inp, out, rescaling=9)


def test_iterations_minimum(tmp_io_dirs):
    inp, out = tmp_io_dirs
    with pytest.raises(ValueError, match="Iterations must be at least 1"):
        make(inp, out, iterations=0)


def test_target_hist_grayscale_shape_and_type(tmp_io_dirs):
    inp, out = tmp_io_dirs
    # as_gray != 0 -> 1D hist of length 256 or 65536 with integer dtype
    ok = make(inp, out, as_gray=1, target_hist=np.ones((256,)))
    assert ok.as_gray == 1
    assert ok.target_hist.shape == (256,)

    with pytest.raises(ValueError):
        make(inp, out, as_gray=1, target_hist=np.ones((128,)))  # wrong length


def test_target_hist_color_shape_and_type(tmp_io_dirs):
    inp, out = tmp_io_dirs
    # as_gray == 0 -> 2D hist (256,3) with integer dtype
    ok = make(inp, out, as_gray=0, target_hist=np.ones((256, 3)))
    assert ok.target_hist.shape == (256, 3)

    with pytest.raises(ValueError, match="must be 2D"):
        make(inp, out, as_gray=0, target_hist=np.ones((256,)))


def test_target_hist_color_shape_and_string(tmp_io_dirs):
    inp, out = tmp_io_dirs
    # as_gray == 0 -> 2D hist (256,3) with integer dtype
    ok = make(inp, out, as_gray=0, target_hist='equal')
    assert ok.target_hist == 'equal'

    with pytest.raises(TypeError, match="string = 'equal'"):
        make(inp, out, as_gray=0, target_hist='none')
