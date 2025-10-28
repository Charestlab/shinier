import shutil, pytest
import numpy as np
from pathlib import Path
from PIL import Image
from shinier import Options, ImageDataset
from shinier.ImageProcessor import ImageProcessor

pytestmark = pytest.mark.unit_tests


def make_rgb(h=8, w=10, seed=0, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return arr.astype(dtype, copy=False)


@pytest.fixture
def simple_dataset(tmp_path):
    inp = tmp_path / "INPUT"; out = tmp_path / "OUTPUT"
    inp.mkdir(); out.mkdir()
    paths = []
    for i in range(2):
        Image.fromarray(make_rgb(seed=i)).save(inp / f"im_{i}.png")
        paths.append(inp / f"im_{i}.png")
    opt = Options(input_folder=inp, output_folder=out, images_format="png",
                  whole_image=1, mode=1, conserve_memory=False, dithering=0, as_gray=0)
    ds = ImageDataset(images=paths, options=opt)
    return ds, opt


def test_init_picks_dataset_options_and_runs(simple_dataset):
    ds, opt = simple_dataset
    # Do NOT pass options explicitly; ImageProcessor should use dataset.options
    proc = ImageProcessor(dataset=ds, options=None, verbose=0)
    assert proc.options is ds.options
    # After init, images should be uint8 in [0,255]
    for im in ds.images:
        assert im.dtype == np.uint8
        assert im.min() >= 0 and im.max() <= 255


def test_uint8_to_float255_converts_and_sets_range(simple_dataset):
    ds, opt = simple_dataset
    proc = ImageProcessor(dataset=ds, options=opt, verbose=0)  # process once
    # Rebuild buffer from current (uint8) images
    buffer = proc.uint8_to_float255(ds.images, ds.buffer)
    for im in buffer:
        assert im.dtype.kind == 'f'
        assert im.min() >= 0.0 and im.max() <= 255.0
    assert hasattr(buffer, "drange") and buffer.drange == (0, 255)


def test_dithering_variants(simple_dataset):
    ds, opt = simple_dataset
    proc = ImageProcessor(dataset=ds, options=opt, verbose=0)
    # Prepare a fresh float buffer in [0,255]
    buf = proc.uint8_to_float255(ds.images, ds.buffer)

    # dithering=0 -> direct uint8 conversion
    out0 = proc.dithering(input_collection=buf, output_collection=ds.images, dithering=0)
    for im in out0:
        assert im.dtype == np.uint8

    # dithering=1 -> noisy bit dithering
    out1 = proc.dithering(input_collection=buf, output_collection=ds.images, dithering=1)
    for im in out1:
        assert im.dtype == np.uint8

    # dithering=2 -> floyd-steinberg
    out2 = proc.dithering(input_collection=buf, output_collection=ds.images, dithering=2)
    for im in out2:
        assert im.dtype == np.uint8


def test_validate_argument_length_mismatch_raises(simple_dataset):
    ds, opt = simple_dataset
    proc = ImageProcessor(dataset=ds, options=opt, verbose=0)
    with pytest.raises(ValueError, match="same size"):
        proc._validate(observed=[1.0, 2.0], expected=[1.0], measures_str=["a", "b"])


def test_hist_spec_seed_set_when_enabled(simple_dataset, monkeypatch):
    ds, opt = simple_dataset
    # Enable hist_specification and choose a mode that triggers seed logic (2,5,6,7,8)
    opt.hist_specification = 1
    opt.mode = 2
    opt.seed = None
    proc = ImageProcessor(dataset=ds, options=opt, verbose=0)
    assert isinstance(proc.seed, int)
    # the log should contain the seed entry (not strictly required to be first)
    assert any("seed=" in str(entry) for entry in proc.log)


def _corr_and_rms(a: np.ndarray, b: np.ndarray):
    a_flat = a.astype(np.float32).ravel()
    b_flat = b.astype(np.float32).ravel()
    # Pearson correlation
    a_mean = a_flat.mean(); b_mean = b_flat.mean()
    num = ((a_flat - a_mean) * (b_flat - b_mean)).sum()
    den = np.sqrt(((a_flat - a_mean)**2).sum() * ((b_flat - b_mean)**2).sum())
    corr = float(num / den) if den != 0 else 1.0
    rms = float(np.sqrt(np.mean((a_flat - b_flat) ** 2)))
    return corr, rms


def _run_processor_and_collect_outputs(input_dir: Path, tmp_path, mode: int, seed: int):
    out = tmp_path / f"OUT_mode{mode}_seed{seed}"
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir()

    # Build fresh dataset from files in input_dir
    paths = sorted(list(input_dir.glob("*.png")))
    opt = Options(input_folder=input_dir, output_folder=out, images_format="png",
                  whole_image=1, mode=mode, conserve_memory=False, as_gray=0,
                  dithering=1,  # enable noisy dithering
                  hist_specification=1,  # enable hist match with noise & seed
                  seed=seed)
    ds = ImageDataset(images=paths, options=opt)
    # ImageProcessor __init__ triggers processing + save_images() for file-backed collections
    _ = ImageProcessor(dataset=ds, options=opt, verbose=0)
    # Collect saved outputs
    outs = [np.array(Image.open(p)) for p in sorted(out.glob("*.png"))]
    return outs


def test_reproducibility_fixed_seed_hist_and_dithering(tmp_path):
    # Make a small input set
    inp1 = tmp_path / "INP1"; inp1.mkdir()
    for i in range(3):
        Image.fromarray(make_rgb(h=16, w=20, seed=i)).save(inp1 / f"im_{i}.png")

    # Run twice with the SAME seed on independent datasets
    seed = 12345
    outs_a = _run_processor_and_collect_outputs(inp1, tmp_path, mode=2, seed=seed)
    outs_b = _run_processor_and_collect_outputs(inp1, tmp_path, mode=2, seed=seed)

    # Compare image-by-image
    for a, b in zip(outs_a, outs_b):
        corr, rms = _corr_and_rms(a, b)
        assert corr > 0.9999999999999999
        assert rms == 0.0
