import warnings

import numpy as np
import pytest
from PIL import Image

from shinier.utils import (
    StimulusMasker,
    betce_gray,
    classic_he_gray,
    compute_ambe,
    compute_bp2bpsim,
    compute_contrast_improvement,
    compute_image_entropy,
    compute_mssim,
    compute_psnr,
    nfldice_gray,
    rdfhe_gray,
    sfcef_gray,
    tidhe_gray,
)

pytestmark = pytest.mark.unit_tests

IE_METHODS = [
    classic_he_gray,
    tidhe_gray,
    rdfhe_gray,
    nfldice_gray,
    betce_gray,
    sfcef_gray,
]
LUT_METHODS = [classic_he_gray, tidhe_gray, rdfhe_gray, nfldice_gray, betce_gray]


@pytest.mark.parametrize("method", IE_METHODS)
def test_image_enhancement_rejects_non_grayscale(method) -> None:
    """Image-enhancement methods are intentionally limited to 2D grayscale images."""
    with pytest.raises(ValueError, match="2D grayscale"):
        method(np.zeros((4, 4, 1), dtype=np.uint8))


@pytest.mark.parametrize("method", IE_METHODS)
def test_image_enhancement_rejects_non_uint8(method) -> None:
    """Image-enhancement methods use the MATLAB uint8 reference domain."""
    with pytest.raises(ValueError, match="dtype uint8"):
        method(np.zeros((4, 4), dtype=np.float64))


@pytest.mark.parametrize("method", IE_METHODS)
def test_image_enhancement_output_shape_dtype_and_range(method) -> None:
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    result = method(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.min(result) >= 0
    assert np.max(result) <= 255


@pytest.mark.parametrize("method", IE_METHODS)
def test_image_enhancement_constant_image_stable(method) -> None:
    result = method(np.full((64, 64), 128, dtype=np.uint8))
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8
    assert np.min(result) >= 0
    assert np.max(result) <= 255


@pytest.mark.parametrize("method", LUT_METHODS)
def test_image_enhancement_mapping_is_monotonic(method) -> None:
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result = method(image)
    assert np.all(np.diff(result.ravel().astype(np.int64)) >= 0)


def test_classic_he_gray_constant_image_unchanged() -> None:
    image = np.full((16, 16), 128, dtype=np.uint8)
    np.testing.assert_array_equal(classic_he_gray(image), image)


def test_classic_he_gray_two_levels_use_full_range() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    image[:, 8:] = 128
    result = classic_he_gray(image)
    assert np.min(result) == 0
    assert np.max(result) == 255


def test_nfldice_gray_midpoint_maps_to_half_range() -> None:
    """With p_l=127.5, level 128 maps just above 0.5*(L-1), matching MATLAB."""
    image = np.full((8, 8), 128, dtype=np.uint8)
    assert int(nfldice_gray(image)[0, 0]) == 129


def test_contrast_metric_ambe_known_values() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)
    shifted = np.full((4, 4), 10, dtype=np.uint8)
    assert compute_ambe(image, image) == pytest.approx(0.0)
    assert compute_ambe(image, shifted) == pytest.approx(10.0)


def test_contrast_metric_ambe_3d_averages_channels() -> None:
    reference = np.zeros((2, 2, 2), dtype=np.float64)
    enhanced = np.zeros_like(reference)
    enhanced[:, :, 0] = 10
    enhanced[:, :, 1] = -10
    assert compute_ambe(reference, enhanced) == pytest.approx(10.0)


def test_contrast_metric_ci_and_entropy_constant_and_varied() -> None:
    constant = np.zeros((4, 4), dtype=np.uint8)
    varied = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    assert compute_contrast_improvement(constant, n_bins=2) == pytest.approx(0.0)
    assert compute_image_entropy(constant, n_bins=2) == pytest.approx(0.0)
    assert compute_contrast_improvement(varied, n_bins=2) > 0
    assert compute_image_entropy(varied, n_bins=2) == pytest.approx(1.0)


def test_contrast_metric_mssim_identical_is_one() -> None:
    image = np.arange(64, dtype=np.float64).reshape(8, 8)
    assert compute_mssim(image, image, data_range=63) == pytest.approx(1.0)


def test_contrast_metric_psnr_known_values() -> None:
    reference = np.zeros((4, 4), dtype=np.float64)
    enhanced = np.ones((4, 4), dtype=np.float64)

    assert compute_psnr(reference, reference, data_range=1) == np.inf
    assert compute_psnr(reference, enhanced, data_range=1) == pytest.approx(0.0)


def test_contrast_metric_psnr_3d_averages_channels() -> None:
    reference = np.zeros((2, 2, 2), dtype=np.float64)
    enhanced = np.zeros_like(reference)
    enhanced[:, :, 0] = 1
    enhanced[:, :, 1] = 2
    expected = (0.0 + (-10.0 * np.log10(4.0))) / 2.0
    assert compute_psnr(reference, enhanced, data_range=1) == pytest.approx(expected)


def test_contrast_metric_bp2bpsim_known_values() -> None:
    reference = np.array([[0]], dtype=np.uint8)
    enhanced = np.array([[1]], dtype=np.uint8)

    assert compute_bp2bpsim(reference, reference) == pytest.approx(1.0)
    assert compute_bp2bpsim(reference, enhanced) == pytest.approx(7.0 / 8.0)
    assert compute_bp2bpsim(reference, enhanced, n_bits=1) == pytest.approx(0.0)


def test_contrast_metric_bp2bpsim_3d_averages_channels() -> None:
    reference = np.zeros((1, 1, 2), dtype=np.uint8)
    enhanced = np.zeros_like(reference)
    enhanced[:, :, 0] = 1
    assert compute_bp2bpsim(reference, enhanced, n_bits=1) == pytest.approx(0.5)


@pytest.mark.parametrize("metric", [compute_ambe, compute_mssim, compute_psnr, compute_bp2bpsim])
def test_pairwise_contrast_metrics_reject_shape_mismatch(metric) -> None:
    with pytest.raises(ValueError, match="same shape"):
        metric(np.zeros((4, 4)), np.zeros((4, 5)))


def test_contrast_metric_bp2bpsim_rejects_invalid_bit_count() -> None:
    with pytest.raises(ValueError, match="n_bits"):
        compute_bp2bpsim(np.zeros((2, 2)), np.zeros((2, 2)), n_bits=0)


def _rgb(value: int, shape: tuple[int, int] = (9, 9)) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.uint8)


def _has_soft_values(values: np.ndarray, low: int = 128, high: int = 255) -> bool:
    return bool(np.any((values > low) & (values < high)))


def test_stimulus_masker_saves_masks_with_expected_dtype_range_and_warning(tmp_path) -> None:
    hard = StimulusMasker(16, cutoff_a=0.6, mask_type="hard")
    soft = StimulusMasker(33, cutoff_a=0.45, mask_type="feathered_disk", edge_width=8)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hard.save_mask(tmp_path / "hard.npy")
    with pytest.warns(RuntimeWarning, match="visualization purposes only"):
        alpha_path = soft.save_mask(tmp_path / "soft")

    alpha = np.load(alpha_path)
    assert alpha_path == tmp_path / "soft.npy"
    assert alpha.shape == (33, 33)
    assert alpha.dtype == np.float32
    assert 0 <= alpha.min() <= alpha.max() <= 1

    with pytest.warns(RuntimeWarning, match="visualization purposes only"):
        valued_path = soft.save_mask(tmp_path / "soft_valued.npy", dtype=np.uint8, inside_value=255, outside_value=128)
    with pytest.warns(RuntimeWarning, match="visualization purposes only"):
        preview_path = soft.save_mask(tmp_path / "soft_preview.png", inside_value=255, outside_value=128)

    valued = np.load(valued_path)
    preview = np.asarray(Image.open(preview_path))
    for saved in (valued, preview):
        values = np.unique(saved)
        assert saved.dtype == np.uint8
        assert 128 in values and 255 in values
        assert _has_soft_values(values)


@pytest.mark.parametrize("background", [128, 0.5])
def test_stimulus_masker_applies_background_and_soft_transition(background) -> None:
    hard = StimulusMasker(9, cutoff_a=0.5, mask_type="hard", background=background, output_dtype=np.uint8)
    masked = hard.apply_mask(_rgb(255))
    np.testing.assert_array_equal(masked[0, 0], [128, 128, 128])
    np.testing.assert_array_equal(masked[4, 4], [255, 255, 255])

    soft = StimulusMasker(33, 0.45, mask_type="feathered_disk", edge_width=8, background=128, output_dtype=np.uint8)
    values = np.unique(soft.apply_mask(_rgb(255, (33, 33)))[:, :, 0])
    assert 128 in values and 255 in values
    assert _has_soft_values(values)


@pytest.mark.parametrize("mask_type,softness", [("feathered_disk", {"edge_width": 8}), ("gaussian", {"sigma": 3})])
def test_stimulus_masker_inward_edge_bias_never_exceeds_hard_footprint(mask_type, softness) -> None:
    common = dict(image_size=65, cutoff_a=0.45, cutoff_b=0.6, offset_a=0.05, offset_b=-0.1)
    hard_exterior = StimulusMasker(**common, mask_type="hard").generate_mask() == 0
    inward = StimulusMasker(**common, mask_type=mask_type, edge_bias="inward", **softness).generate_mask()
    centered = StimulusMasker(**common, mask_type=mask_type, edge_bias="center", **softness).generate_mask()

    np.testing.assert_allclose(inward[hard_exterior], 0.0, atol=1e-6)
    assert centered[hard_exterior].max() > 1e-3


def test_stimulus_masker_apply_accepts_single_batch_mapping_and_grayscale(capsys) -> None:
    masker = StimulusMasker(9, 0.5, mask_type="hard", background=128, output_dtype=np.uint8)
    image, black = _rgb(255), _rgb(0)

    single = masker.apply_mask(image)
    batch = masker.apply_mask([image, black], verbose=False)
    stack = masker.apply_mask(np.stack([image, black]), verbose=False)
    mapping = masker.apply_mask({"face_a.png": image, "face_b.png": black}, verbose=False)

    assert "1 image(s)" in capsys.readouterr().out
    assert isinstance(single, np.ndarray)
    assert [arr.shape for arr in batch] == [image.shape, image.shape]
    assert [arr.shape for arr in stack] == [image.shape, image.shape]
    assert set(mapping) == {"face_a.png", "face_b.png"}
    np.testing.assert_array_equal(batch[0], single)
    np.testing.assert_array_equal(mapping["face_a.png"], single)

    gray = StimulusMasker(9, 0.5, mask_type="hard", output_dtype=np.uint8, preserve_grayscale=True)
    masked_gray = gray.apply_mask(np.full((9, 9), 255, dtype=np.uint8), background=128, verbose=False)
    assert masked_gray.shape == (9, 9)
    assert masked_gray[0, 0] == 128


def test_stimulus_masker_save_masked_stim_paths_logs_and_safety(capsys, tmp_path) -> None:
    image, bad = _rgb(255), _rgb(0, (8, 8))
    masker = StimulusMasker(9, 0.5, mask_type="hard", output_dtype=np.float32)

    single = masker.save_masked_stim(image, tmp_path / "face", background=128, output_dtype=np.uint8)
    paths = masker.save_masked_stim({"face_a.png": image, "face_b.png": _rgb(0)}, tmp_path / "batch", background=128)

    assert capsys.readouterr().out.count("[SHINIER]") == 2
    assert single == tmp_path / "face.png"
    assert [p.name for p in paths] == ["face_a.png", "face_b.png"]
    np.testing.assert_array_equal(np.asarray(Image.open(single))[0, 0], [128, 128, 128])
    assert masker.output_dtype is np.float32

    with pytest.raises(ValueError, match=r"stimulus 'bad_one.png' -- Mask shape"):
        masker.save_masked_stim({"good.png": image, "bad_one.png": bad}, tmp_path)
    with pytest.raises(ValueError, match="output filenames must be unique"):
        masker.save_masked_stim([image, image], tmp_path, names=["same", "same.png"])
    with pytest.raises(ValueError, match="output filenames must be unique"):
        masker.save_masked_stim({"folder/same.png": image, "same.png": image}, tmp_path)


def test_stimulus_masker_defaults_validation_and_error_labels() -> None:
    masker = StimulusMasker(9, 0.5)
    assert masker.mask_type == "hard"
    assert masker.edge_bias == "center"
    np.testing.assert_array_equal(np.unique(masker.generate_mask()), [0.0, 1.0])

    with pytest.raises(ValueError, match=r"Mask shape \(9, 9\) does not match image shape \(8, 8\)"):
        masker.apply_mask(_rgb(0, (8, 8)), verbose=False)
    with pytest.raises(ValueError, match=r"stimulus 'bad_one' -- Mask shape"):
        masker.apply_mask({"good": _rgb(0), "bad_one": _rgb(0, (8, 8))}, verbose=False)
    with pytest.raises(ValueError, match=r"stimulus 1 -- Mask shape"):
        masker.apply_mask([_rgb(0), _rgb(0, (8, 8))], verbose=False)


@pytest.mark.parametrize(
    ("mask_type", "kwargs", "expected", "unexpected"),
    [
        ("hard", {}, ("mask_type=hard",), ("edge_bias", "sigma=", "edge_width=")),
        ("gaussian", {"sigma": 3}, ("mask_type=gaussian", "edge_bias=", "sigma=3"), ("edge_width=",)),
        ("feathered_disk", {"edge_width": 4}, ("mask_type=feathered_disk", "edge_bias=", "edge_width=4"), ("sigma=",)),
    ],
)
def test_stimulus_masker_verbose_logs_relevant_params(capsys, mask_type, kwargs, expected, unexpected) -> None:
    StimulusMasker(9, 0.5, cutoff_b=0.6, mask_type=mask_type, **kwargs).apply_mask(_rgb(0))
    out = capsys.readouterr().out
    assert "1 image(s)" in out and "cutoff_a=0.5" in out
    assert all(token in out for token in expected)
    assert not any(token in out for token in unexpected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image_size": 0, "cutoff_a": 0.5}, "image_size"),
        ({"image_size": 9, "cutoff_a": 0}, "cutoff_a"),
        ({"image_size": 9, "cutoff_a": 0.5, "cutoff_b": 0}, "cutoff_b"),
        ({"image_size": 9, "cutoff_a": 0.5, "sigma": -1}, "sigma"),
        ({"image_size": 9, "cutoff_a": 0.5, "edge_width": -1}, "edge_width"),
        ({"image_size": 9, "cutoff_a": 0.5, "mask_type": "bad"}, "mask_type"),
        ({"image_size": 9, "cutoff_a": 0.5, "edge_bias": "bad"}, "edge_bias"),
    ],
)
def test_stimulus_masker_rejects_invalid_parameters(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        StimulusMasker(**kwargs)


@pytest.mark.parametrize(
    ("original", "expected_type", "tolerance"),
    [
        (
            StimulusMasker((129, 161), 0.52, cutoff_b=0.74, offset_a=0.08, offset_b=-0.06, mask_type="hard"),
            "hard",
            0.01,
        ),
        (
            StimulusMasker(
                129,
                0.52,
                cutoff_b=0.74,
                offset_a=0.08,
                offset_b=-0.06,
                mask_type="feathered_disk",
                edge_width=7,
            ),
            "feathered_disk",
            0.02,
        ),
    ],
)
def test_stimulus_masker_from_mask_recovers_parameters(original, expected_type, tolerance) -> None:
    fitted, error = StimulusMasker.from_mask(original.generate_mask(), return_error=True)
    assert fitted.mask_type == expected_type
    assert fitted.cutoff_a == pytest.approx(original.cutoff_a, abs=tolerance)
    assert fitted.cutoff_b == pytest.approx(original.cutoff_b, abs=tolerance)
    assert fitted.offset_a == pytest.approx(original.offset_a, abs=tolerance)
    assert fitted.offset_b == pytest.approx(original.offset_b, abs=tolerance)
    assert error < 0.01
    if expected_type == "feathered_disk":
        assert fitted.edge_width == pytest.approx(original.edge_width, abs=0.5)


def test_stimulus_masker_from_mask_loads_npy_and_rejects_constant_mask(tmp_path) -> None:
    original = StimulusMasker(33, 0.5, cutoff_b=0.7, offset_b=0.1, mask_type="hard")
    path = tmp_path / "mask.npy"
    np.save(path, original.generate_mask())

    fitted = StimulusMasker.from_mask(path)
    assert fitted.mask_type == "hard"
    assert fitted.image_size == (33, 33)
    assert fitted.cutoff_a == pytest.approx(original.cutoff_a, abs=0.05)
    assert fitted.cutoff_b == pytest.approx(original.cutoff_b, abs=0.05)
    assert fitted.offset_b == pytest.approx(original.offset_b, abs=0.05)

    with pytest.raises(ValueError, match="at least two finite values"):
        StimulusMasker.from_mask(np.ones((8, 8)))


def test_stimulus_masker_from_interactive_mask_builds_from_image(monkeypatch) -> None:
    def fake_interactive_mask(self, _image):
        self.offset_b = 0.25
        return self.generate_mask()

    monkeypatch.setattr(StimulusMasker, "interactive_mask", fake_interactive_mask)
    masker = StimulusMasker.from_interactive_mask(np.zeros((8, 9, 3), dtype=np.uint8), cutoff_a=0.6)
    assert masker.image_size == (8, 9)
    assert masker.cutoff_a == 0.6
    assert masker.offset_b == 0.25


def test_stimulus_masker_interactive_mask_widgets(monkeypatch) -> None:
    # Backend is forced to Agg in conftest.py, before pyplot is ever imported anywhere
    # in the session -- do not call matplotlib.use() here (see conftest.py for why).
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    created = {"Button": [], "TextBox": []}
    for name in created:
        original = getattr(widgets, name)

        def tracker_class(original, name):
            class Tracker(original):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    created[name].append(self)
            return Tracker

        monkeypatch.setattr(widgets, name, tracker_class(original, name))

    masker = StimulusMasker(32, 0.5, cutoff_b=0.6, mask_type="feathered_disk", edge_width=5)
    masker.interactive_mask(np.zeros((32, 32, 3), dtype=np.uint8))

    cutoff_a_textbox = created["TextBox"][1]
    edge_button = next(button for button in created["Button"] if "Edge" in button.label.get_text())
    reset_button = next(button for button in created["Button"] if button.label.get_text() == "Reset")

    cutoff_a_textbox.set_val("0.9")
    cutoff_a_textbox._observers.process("submit", cutoff_a_textbox.text)
    assert masker.cutoff_a == pytest.approx(0.9)

    cutoff_a_textbox.set_val("99")
    cutoff_a_textbox._observers.process("submit", cutoff_a_textbox.text)
    assert masker.cutoff_a == pytest.approx(1.5)

    edge_button._observers.process("clicked", None)
    assert masker.edge_bias == "inward"

    reset_button._observers.process("clicked", None)
    assert masker.cutoff_a == pytest.approx(0.5)
    assert masker.edge_bias == "center"
    assert masker.mask_type == "feathered_disk"
    assert cutoff_a_textbox.text == "0.500"
