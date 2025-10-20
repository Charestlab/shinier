
import numpy as np
import pytest
from PIL import Image
from shinier import ImageDataset, Options

pytestmark = pytest.mark.unit_tests


def make_rgb(h=8, w=10, seed=0, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return arr.astype(dtype, copy=False)


@pytest.fixture
def imgs_dir(tmp_path):
    d = tmp_path / "INPUT"
    d.mkdir()
    paths = []
    for i in range(3):
        arr = make_rgb(seed=i)
        p = d / f"im_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return d, paths


@pytest.fixture
def masks_dir(tmp_path):
    d = tmp_path / "MASKS"
    d.mkdir()
    return d


def test_init_with_explicit_images_and_options(imgs_dir, tmp_path):
    inp, paths = imgs_dir
    out = tmp_path / "OUTPUT"
    out.mkdir()
    opt = Options(input_folder=inp, output_folder=out, images_format="png", conserve_memory=True, as_gray=0, whole_image=1, mode=1)
    ds = ImageDataset(images=paths, options=opt)
    assert ds.n_images == 3
    assert ds.images_name == [p.name for p in paths]
    # channels detected
    assert getattr(ds.images, "n_channels", None) in (1, 3)


def test_init_loads_from_options_when_images_none(imgs_dir, tmp_path):
    inp, _ = imgs_dir
    out = tmp_path / "OUTPUT"
    out.mkdir()
    opt = Options(input_folder=inp, output_folder=out, images_format="png", conserve_memory=False, as_gray=0, whole_image=1, mode=1)
    ds = ImageDataset(images=None, options=opt)
    assert ds.n_images == 3
    assert all(n.endswith(".png") for n in ds.images_name)


def test_validate_raises_if_single_image(tmp_path):
    inp = tmp_path / "INPUT"; out = tmp_path / "OUTPUT"
    inp.mkdir(); out.mkdir()
    # only one image
    arr = make_rgb(seed=0)
    Image.fromarray(arr).save(inp / "one.png")
    opt = Options(input_folder=inp, output_folder=out, images_format="png", whole_image=1, mode=1)
    with pytest.raises(ValueError, match="More than one image"):
        ImageDataset(images=None, options=opt)


def test_masks_count_must_be_1_or_equal_to_images(imgs_dir, masks_dir, tmp_path):
    inp, paths = imgs_dir
    out = tmp_path / "OUTPUT"; out.mkdir()
    # create 2 masks (invalid since we have 3 images)
    for i in range(2):
        Image.fromarray(np.full((8,10), 255, np.uint8)).save(masks_dir / f"mask_{i}.png")
    opt = Options(input_folder=inp, output_folder=out, images_format="png", whole_image=3, masks_folder=masks_dir, masks_format="png", mode=1)
    with pytest.raises(ValueError, match="number of masks"):
        ImageDataset(images=None, options=opt)


def test_masks_shape_must_match_images(imgs_dir, masks_dir, tmp_path):
    inp, paths = imgs_dir
    out = tmp_path / "OUTPUT"; out.mkdir()
    # create one mask with different shape (valid count=1 but wrong size)
    Image.fromarray(np.full((7,9), 255, np.uint8)).save(masks_dir / "mask.png")
    opt = Options(input_folder=inp, output_folder=out, images_format="png", whole_image=3, masks_folder=masks_dir, masks_format="png", mode=1)
    with pytest.raises(ValueError, match="same shape"):
        ImageDataset(images=None, options=opt)


def test_mode_ge3_creates_magnitude_phase_placeholders(imgs_dir, tmp_path):
    inp, paths = imgs_dir
    out = tmp_path / "OUTPUT"; out.mkdir()
    opt = Options(input_folder=inp, output_folder=out, images_format="png", whole_image=1, mode=3, conserve_memory=True)
    ds = ImageDataset(images=None, options=opt)
    assert hasattr(ds, "magnitudes") and hasattr(ds, "phases")
    assert len(ds.magnitudes) == ds.n_images
    assert len(ds.phases) == ds.n_images


def test_save_images_writes_to_output(imgs_dir, tmp_path):
    inp, _ = imgs_dir
    out = tmp_path / "OUTPUT"; out.mkdir()
    opt = Options(input_folder=inp, output_folder=out, images_format="png", whole_image=1, mode=1, conserve_memory=True)
    ds = ImageDataset(images=None, options=opt)
    ds.save_images()
    # should have saved 3 files
    saved = list(out.glob("*.png")) + list(out.glob("*.npy"))
    assert len(saved) >= 3


def test_close_does_not_raise(imgs_dir, tmp_path):
    inp, _ = imgs_dir
    out = tmp_path / "OUTPUT"; out.mkdir()
    opt = Options(input_folder=inp, output_folder=out, images_format="png", whole_image=1, mode=1, conserve_memory=True)
    ds = ImageDataset(images=None, options=opt)
    ds.close()  # should not raise


def test_buffer_is_instantiated_when_mode_ge1(imgs_dir, tmp_path):
    inp, _ = imgs_dir
    out = tmp_path / "OUTPUT"; out.mkdir()
    # mode=1 should create buffer according to the provided design
    opt = Options(input_folder=inp, output_folder=out, images_format="png",
                  whole_image=1, mode=1, conserve_memory=True)
    ds = ImageDataset(images=None, options=opt)
    assert hasattr(ds, "buffer") and ds.buffer is not None
    assert len(ds.buffer) == ds.n_images
    # buffer elements are boolean arrays (shape of images)
    b0 = ds.buffer[0]
    assert b0.dtype == np.bool_ and b0.shape[:2] == ds.images[0].shape[:2]
    ds.close()


def test_magnitude_phase_created_when_mode_ge3(imgs_dir, tmp_path):
    inp, _ = imgs_dir
    out = tmp_path / "OUTPUT"; out.mkdir()
    # mode=3 should create magnitudes & phases
    opt = Options(input_folder=inp, output_folder=out, images_format="png",
                  whole_image=1, mode=3, conserve_memory=True)
    ds = ImageDataset(images=None, options=opt)
    assert hasattr(ds, "magnitudes") and ds.magnitudes is not None
    assert hasattr(ds, "phases") and ds.phases is not None
    assert len(ds.magnitudes) == ds.n_images
    assert len(ds.phases) == ds.n_images
    ds.close()


def test_magnitude_phase_absent_when_mode_eq9(imgs_dir, tmp_path):
    inp, _ = imgs_dir
    out = tmp_path / "OUTPUT"; out.mkdir()
    # mode=9 should NOT create magnitudes & phases
    opt = Options(input_folder=inp, output_folder=out, images_format="png",
                  whole_image=1, mode=9, conserve_memory=True)
    ds = ImageDataset(images=None, options=opt)
    assert not hasattr(ds, "magnitudes") or ds.magnitudes is None
    assert not hasattr(ds, "phases") or ds.phases is None
    ds.close()