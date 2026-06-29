"""Unit coverage for parameters collapsed in the pruned validation mode.

Each test exercises one value that the pruned mode skips, verifying that it runs
without error and produces finite output. See tests/README.md for rationale.
"""
import numpy as np
import pytest
from pathlib import Path
from shinier import ImageDataset, Options, ImageProcessor
from tests import utils as utils_test

pytestmark = pytest.mark.unit_tests

_BASE_OPTS = dict(
    mode=2,
    as_gray=False,
    linear_luminance=False,
    rec_standard=2,
    legacy_mode=False,
    iterations=1,
    hist_optim=False,
    hist_specification=1,
    rescaling=2,
    dithering=0,
    conserve_memory=False,
    fft_padding_mode=0,
    standalone_op="ie_methods",
    ie_methods="tidhe",
    seed=None,
    verbose=-1,
)


def _run(tmp_path: Path, **overrides) -> ImageProcessor:
    tmp_path.mkdir(parents=True, exist_ok=True)
    buffers = utils_test.prepare_images(utils_test.IMAGE_PATH)
    images = utils_test.select_n_imgs(buffers["images"], n=2, seed=0)
    opts = Options(**{**_BASE_OPTS, "output_folder": tmp_path, **overrides})
    ds = ImageDataset.model_construct(images=images, options=opts)
    return ImageProcessor.model_construct(dataset=ds, options=opts, verbose=-1, from_validation_test=True)


# ----- ie_methods (pruned keeps only "tidhe") -----

@pytest.mark.parametrize("method", ["classic_he", "rdfhe", "nfldice", "betce", "sfcef"])
def test_ie_method_runs(method, tmp_path):
    proc = _run(tmp_path, mode=9, standalone_op="ie_methods", ie_methods=method)
    for img in proc._final_buffer:
        assert np.all(np.isfinite(img)), f"{method}: NaN/Inf in output"


# ----- fft_padding_mode (pruned keeps 0 and 1) -----

@pytest.mark.parametrize("padding_mode", [2, 3])
def test_fft_padding_mode_runs(padding_mode, tmp_path):
    proc = _run(tmp_path, mode=3, fft_padding_mode=padding_mode)
    for img in proc._final_buffer:
        assert np.all(np.isfinite(img)), f"fft_padding_mode={padding_mode}: NaN/Inf in output"


# ----- rec_standard (pruned keeps only 1 = rec601) -----

@pytest.mark.parametrize("rs", [2, 3])
def test_rec_standard_runs(rs, tmp_path):
    proc = _run(tmp_path, rec_standard=rs)
    assert proc._final_buffer is not None


# ----- dithering (pruned keeps 0 and 1) -----

def test_dithering_2_floyd_steinberg_runs(tmp_path):
    proc = _run(tmp_path, dithering=2, as_gray=True)
    for img in proc._final_buffer:
        assert np.all(np.isfinite(img))


# ----- hist_specification (pruned keeps 1 and 3) -----

@pytest.mark.parametrize("hs", [2, 4])
def test_hist_specification_runs(hs, tmp_path):
    proc = _run(tmp_path, mode=2, hist_specification=hs)
    assert proc._final_buffer is not None


# ----- rescaling (pruned keeps 0 and 2) -----

@pytest.mark.parametrize("rescaling", [1, 3])
def test_rescaling_runs(rescaling, tmp_path):
    proc = _run(tmp_path, mode=3, rescaling=rescaling)
    for img in proc._final_buffer:
        assert np.all(np.isfinite(img)), f"rescaling={rescaling}: NaN/Inf in output"


# ----- conserve_memory (pruned keeps False) -----

def test_conserve_memory_true_matches_false(tmp_path):
    proc_off = _run(tmp_path / "off", conserve_memory=False, seed=12345)
    proc_on = _run(tmp_path / "on", conserve_memory=True, seed=12345)
    for a, b in zip(proc_off._final_buffer, proc_on._final_buffer):
        np.testing.assert_allclose(a, b, atol=1e-6)


# ----- seed (pruned keeps None) -----

def test_seeded_run_is_reproducible(tmp_path):
    proc1 = _run(tmp_path / "r1", mode=2, hist_specification=2, seed=9999)
    proc2 = _run(tmp_path / "r2", mode=2, hist_specification=2, seed=9999)
    for a, b in zip(proc1._final_buffer, proc2._final_buffer):
        np.testing.assert_array_equal(a, b)
