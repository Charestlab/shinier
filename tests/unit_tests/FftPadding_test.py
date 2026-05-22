import numpy as np
import pytest

from shinier import ImageDataset, ImageProcessor, Options
from shinier.utils import DEFAULT_FFT_PADDING_RATIO, _crop_after_fft, _pad_for_fft, image_spectrum


pytestmark = pytest.mark.unit_tests


def _make_gray(h: int = 24, w: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def test_fft_padding_helpers_shapes_and_values() -> None:
    image = np.arange(20, dtype=np.float64).reshape(4, 5) / 19
    assert _pad_for_fft(image, None) is image

    pad = int(min(image.shape) * DEFAULT_FFT_PADDING_RATIO)
    padded = _pad_for_fft(image, "reflect")
    assert padded.shape == (image.shape[0] + 2 * pad, image.shape[1] + 2 * pad)
    assert _crop_after_fft(padded, image.shape).shape == image.shape

    rgb = np.zeros((5, 6, 3), dtype=np.float64)
    rgb[..., 0] = 0.25
    pad = int(min(rgb.shape[:2]) * DEFAULT_FFT_PADDING_RATIO)
    padded_constant = _pad_for_fft(rgb, "constant", value=128)
    assert padded_constant.shape == (rgb.shape[0] + 2 * pad, rgb.shape[1] + 2 * pad, 3)
    assert np.isclose(padded_constant[0, 0, 0], 128 / 255)

    padded_mean = _pad_for_fft(rgb, "constant")
    assert np.isclose(padded_mean[0, 0, 0], rgb.mean())


def test_image_spectrum_padding_preserves_default_shape() -> None:
    image = np.zeros((5, 6, 3), dtype=np.float64)
    mag_default, phase_default = image_spectrum(image, rescale=False)
    mag_padded, phase_padded = image_spectrum(image, rescale=False, fft_padding_mode="symmetric")
    pad = int(min(image.shape[:2]) * DEFAULT_FFT_PADDING_RATIO)

    assert mag_default.shape == (5, 6, 3)
    assert phase_default.shape == (5, 6, 3)
    assert mag_padded.shape == (image.shape[0] + 2 * pad, image.shape[1] + 2 * pad, 3)
    assert phase_padded.shape == (image.shape[0] + 2 * pad, image.shape[1] + 2 * pad, 3)


def test_fft_padding_options_validation(tmp_path) -> None:
    opt = Options(output_folder=tmp_path, fft_padding_mode="constant", fft_padding_value=128)
    assert opt.fft_padding_mode == "constant"
    assert opt.fft_padding_value == 128

    with pytest.raises(ValueError):
        Options(output_folder=tmp_path, fft_padding_mode="constant", fft_padding_value=300)


@pytest.mark.parametrize("mode,padding", [(3, "constant"), (4, "reflect")])
def test_fft_padding_preserves_processed_image_size(tmp_path, mode, padding) -> None:
    images = [_make_gray(seed=1), _make_gray(seed=2)]
    opt = Options(
        output_folder=tmp_path,
        mode=mode,
        as_gray=True,
        verbose=-1,
        fft_padding_mode=padding,
    )
    dataset = ImageDataset(images=images, options=opt)
    processor = ImageProcessor(dataset=dataset, options=opt, verbose=-1)

    assert processor._final_buffer[0].shape == images[0].shape
    assert processor._target_spectrum.shape[0] > images[0].shape[0]
    assert processor._target_spectrum.shape[1] > images[0].shape[1]


def test_direct_target_spectrum_must_match_padded_shape(tmp_path) -> None:
    images = [_make_gray(20, 22, seed=3), _make_gray(20, 22, seed=4)]
    target_spectrum = np.ones((20, 22), dtype=np.float64)
    opt = Options(
        output_folder=tmp_path,
        mode=4,
        as_gray=True,
        verbose=-1,
        target_spectrum=target_spectrum,
        fft_padding_mode="reflect",
    )
    dataset = ImageDataset(images=images, options=opt)

    with pytest.raises(RuntimeError, match="padded FFT shape"):
        ImageProcessor(dataset=dataset, options=opt, verbose=-1)
