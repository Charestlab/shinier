import numpy as np
import pytest
from shinier import ImageListIO

pytestmark = pytest.mark.unit_tests


def make_rgb(h=8, w=10, seed=0, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return arr.astype(dtype, copy=False)


@pytest.fixture
def tmp_img_dir(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    from PIL import Image
    ims = []
    for i in range(3):
        arr = make_rgb(seed=i)
        p = d / f"im_{i}.png"
        Image.fromarray(arr).save(p)
        ims.append(p)
    return d, ims


def test_init_from_glob_pattern(tmp_img_dir):
    d, ims = tmp_img_dir
    coll = ImageListIO(str(d / "*.png"), conserve_memory=True, as_gray=0)
    assert len(coll) == 3
    a0 = coll[0]
    assert a0.ndim == 3 and a0.shape[2] == 3


def test_init_from_directory_path(tmp_img_dir):
    d, ims = tmp_img_dir
    coll = ImageListIO(d, conserve_memory=False, as_gray=0)
    assert len(coll) == 3
    assert coll[1].shape[:2] == (8, 10)


def test_init_from_list_of_paths(tmp_img_dir):
    d, ims = tmp_img_dir
    coll = ImageListIO(ims, conserve_memory=True, as_gray=2)
    x = coll[2]
    assert x.ndim == 2 and x.dtype == np.uint8


def test_init_from_list_of_arrays_conserve_memory_true(tmp_path):
    arrays = [make_rgb(seed=s) for s in range(4)]
    coll = ImageListIO(arrays, conserve_memory=True, as_gray=0)
    assert len(coll) == 4
    tmp_dir = getattr(coll, "_temp_dir", None)
    assert tmp_dir is not None and tmp_dir.exists()
    assert len(list(tmp_dir.glob("*.npy"))) == 4
    assert np.array_equal(coll[1], arrays[1])


def test_init_from_list_of_arrays_as_gray(tmp_path):
    arrays = [make_rgb(seed=42), make_rgb(seed=43)]
    coll = ImageListIO(arrays, conserve_memory=False, as_gray=1)
    im = coll[0]

    assert im.ndim == 2 and im.dtype == np.uint8


def test_invalid_as_gray_raises(tmp_img_dir):
    d, _ = tmp_img_dir
    with pytest.raises(ValueError, match="as_gray.*0.*1.*2"):
        ImageListIO(d, as_gray=5)


def test_mixed_input_types_raise(tmp_img_dir):
    d, ims = tmp_img_dir
    mixed = [ims[0], make_rgb(seed=1)]
    with pytest.raises(TypeError):
        ImageListIO(mixed)


def test_glob_no_match_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        ImageListIO(str(d / "*.png"))


def test_getitem_bounds_and_negative_indexing(tmp_img_dir):
    d, ims = tmp_img_dir
    coll = ImageListIO(ims)
    # negative indexing is allowed
    last = coll[-1]
    assert last is not None
    # too positive
    with pytest.raises(IndexError):
        _ = coll[len(coll)]
    # too negative
    with pytest.raises(IndexError):
        _ = coll[-len(coll)-1]


def test_setitem_writable_updates_data_and_persistence_when_conserve_true(tmp_path):
    arrays = [make_rgb(seed=s) for s in range(2)]
    coll = ImageListIO(arrays, conserve_memory=True, as_gray=0)
    new_im = make_rgb(h=8, w=10, seed=999)
    coll[0] = new_im
    # Should reflect on read
    got = coll[0]
    assert np.array_equal(got, new_im)
    # Should exist in temp storage
    tmp_dir = getattr(coll, "_temp_dir", None)
    assert tmp_dir is not None and any(tmp_dir.glob("*.npy"))


def test_readonly_copy_blocks_setitem(tmp_img_dir):
    d, ims = tmp_img_dir
    coll = ImageListIO(ims, conserve_memory=False, as_gray=0)
    ro = coll.readonly_copy()
    with pytest.raises(RuntimeError, match="read-only|read only|not supported"):
        ro[0] = coll[0]


def test_final_save_all_writes_files(tmp_path):
    arrays = [make_rgb(6, 7, seed=s) for s in range(3)]
    out_dir = tmp_path / "OUT"
    coll = ImageListIO(arrays, conserve_memory=True, as_gray=0, save_dir=out_dir)
    coll.final_save_all()
    saved = list(out_dir.glob("*"))
    assert len(saved) == 3
    tmp_dir = getattr(coll, "_temp_dir", None)
    assert tmp_dir is None or not tmp_dir.exists()


def test_close_cleans_temp_dir(tmp_path):
    arrays = [make_rgb(seed=s) for s in range(2)]
    coll = ImageListIO(arrays, conserve_memory=True)
    tmp_dir = getattr(coll, "_temp_dir", None)
    assert tmp_dir is not None and tmp_dir.exists()
    coll.close()
    assert getattr(coll, "_temp_dir", None) is None or not tmp_dir.exists()
