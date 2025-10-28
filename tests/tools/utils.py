# tests/_helpers.py
from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, Literal
import pickle

import numpy as np
from PIL import Image

from shinier import utils


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Mapping for utils.rgb2gray's conversion_type (when precomputing targets)
AS_GRAY_NAME = {0: None, 1: "equal", 2: "rec601", 3: "rec709", 4: "rec2020"}

# Get Image path
IMAGE_PATH = Path(__file__).resolve().parent.parent / 'IMAGES/SAMPLE_64X64/'

ComboType = Tuple[
    int,  # mode
    int,  # whole_image
    int,  # dithering
    int,  # as_gray
    int,  # hist_specification
    int,  # hist_optim
    int,  # rescaling
    bool,  # safe_lum_match
    Tuple[int, int],  # target_lum
    str,  # target_hist_choice ("target" | "none")
    str,  # target_spec_choice ("target" | "none")
]

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    """Ensure all paths exist, or raise.

    Args:
        paths: Paths to check.

    Raises:
        FileNotFoundError: If any path does not exist.
    """
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")


def coerce_to_rgb(paths: List[Path], scratch_dir: Path) -> List[Path]:
    """Convert images to RGB and write copies into `scratch_dir`.

    Useful when originals have alpha channels or are not RGB.

    Args:
        paths: Source image paths.
        scratch_dir: Destination directory for coerced images.

    Returns:
        List of coerced image paths (in `scratch_dir`).
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for p in paths:
        with Image.open(p) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            q = scratch_dir / p.name
            im.save(q)
            out.append(q)
    return out


def save_pickle(obj: Any, filename: Union[str, Path]) -> None:
    """Serialize a Python object to a pickle file.

    Args:
        obj: The Python object to serialize.
        filename: Path to the file where the object will be saved.
    """
    path = Path(filename)
    with path.open("wb") as output_file:
        pickle.dump(obj, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: Union[str, Path]) -> Any:
    """Deserialize a Python object from a pickle file.

    Args:
        filename: Path to the pickle file to load.

    Returns:
        The deserialized Python object.
    """
    path = Path(filename)
    with path.open("rb") as input_file:
        return pickle.load(input_file)


def save_json(obj: Any, filename: Union[str, Path], *, indent: int = 2) -> None:
    """Serialize a Python object to a JSON file.

    Args:
        obj: The Python object to serialize. Must be JSON-serializable.
        filename: Path to the file where the object will be saved.
        indent: Number of spaces to use for indentation in the JSON file. Defaults to 2.
    """
    path = Path(filename)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(obj, output_file, indent=indent, ensure_ascii=False)


def load_json(filename: Union[str, Path]) -> Any:
    """Deserialize a Python object from a JSON file.

    Args:
        filename: Path to the JSON file to load.

    Returns:
        The deserialized Python object.
    """
    path = Path(filename)
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def dump_failure_context(combo_dict: dict, rec: dict, tmp_root: Path, seed: int, selected_paths: list[Path], file_type: Literal['json', 'pkl'] = 'json') -> Path:
    """Dump combo and record context to a JSON or PKL file for easier reproduction.

    Args:
        combo_dict: Options dictionary for the failed combo.
        rec: The failing validation record.
        tmp_root: Base temporary folder.
        seed: For reproducibility
        selected_paths: Images that were selected.
        file_type: Either json (default) or pkl

    Returns:
        Path to the dumped file.
    """
    if file_type.lower() not in ['pkl', 'json']:
        raise ValueError('file_type should be either pkl or json (default).')
    dump_path = tmp_root / f"failure_{uuid.uuid4().hex[:8]}.{file_type}"
    cleaned_rec = rec.copy()

    # Strip ANSI codes if log_result present
    if 'log_result' in cleaned_rec:
        cleaned_rec['log_result_clean'] = strip_ansi(str(cleaned_rec['log_result']))

    payload = {
        "combo_opts_kwargs": {k: str(v) if isinstance(v, Path) else v for k, v in combo_dict.items()},
        "seed": int(seed),
        "selected_images": [str(p) for p in selected_paths],
        "record": cleaned_rec,  # include the failing validation record (already JSON-safe via default=str earlier)
    }

    if file_type.lower() == 'pkl':
        save_pickle(payload, dump_path)
    else:
        save_json(payload, dump_path)
    return dump_path


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences for cleaner debug output."""
    return ANSI_ESCAPE.sub('', text)


def deterministic_seed_from_combo(
        combo: ComboType,
        *,
        salt: str = "shinier/validation:v1",
) -> int:
    """Return a stable 32-bit seed derived from an options combo.

    Args:
      combo: Tuple of only JSON-serializable primitives (ints, bools, strs, small tuples).
      salt:  Versioned namespace so you can change encoding without breaking past seeds.

    Returns:
      A 32-bit integer suitable for numpy.random.Generator.
    """

    def enc(x) -> str:
        if isinstance(x, bool):
            return f"b:{int(x)};"
        if isinstance(x, int):
            return f"i:{x};"
        if isinstance(x, str):
            return f"s:{x};"
        if isinstance(x, tuple):
            return "t:[" + "".join(enc(e) for e in x) + "];"
        # Fallback (avoid using repr(Path) etc.; keep it deterministic across platforms)
        return f"u:{str(x)};"

    msg = salt + "|" + enc(combo)
    h = hashlib.sha256(msg.encode("utf-8")).digest()

    # Use first 4 bytes → [0, 2**32-1]
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def get_small_imgs_path(dirpath: Path) -> List[Path]:
    """Get images' path from a directory.

    Args:
        dirpath: Directory containing the images.

    Returns:
        A list of paths corresponding to the randomly selected images.

    Raises:
        FileNotFoundError: If dirpath does not exist.
    """
    if not dirpath.exists():
        raise FileNotFoundError(f"{dirpath} does not exist.")

    # Collect all image file paths (common formats)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    return sorted(p for p in dirpath.iterdir() if p.suffix.lower() in extensions)


def select_n_imgs(all_paths: List[Path], n: int = 2, seed: int = 0) -> List[Path]:
    """Randomly select ``n`` image paths without replacement.

    Args:
        all_paths: List of image paths to sample from.
        n: Number of paths to select (1 ≤ n ≤ len(all_paths)).
        seed: Seed for reproducibility.

    Returns:
        A list of ``n`` distinct paths.

    Raises:
        ValueError: If ``n`` is outside the valid range.
    """
    total = len(all_paths)
    if not (1 <= n <= total):
        raise ValueError(
            f"n must be in [1, {total}], got n={n} with {total} available images."
        )
    rng = np.random.default_rng(seed)
    return list(rng.choice(all_paths, size=n, replace=False))


def make_imgs(dirpath: Path, h: int = 64, w: int = 64, n: int = 2, seed: int = 0) -> None:
    """Create n random RGB images of size HxW in dirpath.

    Args:
        dirpath: Output directory.
        h (int): Image height
        w (int): Image width
        n: Number of images to create.
        seed: RNG seed for reproducibility.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(dirpath / f"im_{i}.png")


def make_masks(dirpath: Path, h: int = 64, w: int = 64, n: int = 1) -> None:
    """Create n binary ellipse masks of size HxW in dirpath.

    Args:
        dirpath: Output directory.
        h (int): Image height
        w (int): Image width
        n: Number of masks to create.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    mask_bool = ((yy - cy) ** 2) / (0.6 * cy ** 2) + ((xx - cx) ** 2) / (0.6 * cx ** 2) <= 1.0
    mask = (mask_bool.astype(np.uint8) * 255)
    for i in range(n):
        Image.fromarray(mask).save(dirpath / (f"mask_{i}.png" if n > 1 else "mask.png"))


def precompute_targets(src_img: np.ndarray) -> Dict[str, Dict[int, np.ndarray]]:
    """Precompute target histogram/spectrum for all as_gray modes using utils.

    For each `as_gray` ∈ {0,1,2,3,4}:
      - "hist"[ag] := utils.imhist(image_in_that_space)
      - "spec"[ag] := magnitude spectrum from utils.image_spectrum(image_in_that_space)[0]

    Args:
        src_img: Reference RGB image (H, W, 3) uint8.

    Returns:
        Dict with:
          - "hist": {ag: hist_ndarray}
          - "spec": {ag: spectrum_mag_ndarray}
    """
    out = {"hist": {}, "spec": {}}

    # Color space (as_gray == 0)
    out["hist"][0] = utils.imhist(src_img)  # (256, 3)
    out["spec"][0], _ = utils.image_spectrum(src_img)

    # Grayscale variants (as_gray in {1,2,3,4})
    for ag in (1, 2, 3, 4):
        conv = AS_GRAY_NAME[ag]
        g = utils.rgb2gray(src_img, conversion_type=conv)
        out["hist"][ag] = utils.imhist(g)       # (256,)
        out["spec"][ag], _ = utils.image_spectrum(g)

    return out
