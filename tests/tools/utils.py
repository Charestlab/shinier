# tests/_helpers.py
from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple, Union, Literal, Optional, get_args
import pickle
import copy
import numpy as np
from PIL import Image
from shinier import utils
from shinier.ImageListIO import ImageListIO
from shinier.color import ColorTreatment, REC_STANDARD, RGB_STANDARD

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Mapping for utils.rgb2gray's conversion_type (when precomputing targets)
AS_GRAY_NAME = {0: None, 1: "equal", 2: "rec601", 3: "rec709", 4: "rec2020"}

# Get Image path
IMAGE_PATH = Path(__file__).resolve().parent.parent / 'assets/SAMPLE_64X64/'

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
DB_PATH = Path(__file__).resolve().parent.parent / "hash_registry.db"


# ---------------------------------------------------------------------
# SQLite-backed HASH registry
# ---------------------------------------------------------------------
def combo_hash(combo: Dict) -> str:
    """Return a short, deterministic hash for a given Options combo."""
    combo_serialized = json.dumps(combo, sort_keys=True, default=str)
    return hashlib.sha1(combo_serialized.encode()).hexdigest()


def _ensure_db() -> None:
    """Create the SQLite registry if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hashes (
                hash TEXT PRIMARY KEY,
                status TEXT DEFAULT 'pending',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                error TEXT
            )
            """
        )
        conn.commit()


def mark_hash_range_done(start: int, end: int) -> None:
    """Mark all hashes whose integer value is in [start, end] as 'done'.
    Args:
        start: Lowest integer hash value (inclusive).
        end: Highest integer hash value (inclusive).
    """
    _ensure_db()
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            """
            UPDATE hashes
            SET status = 'done',
                error = NULL,
                timestamp = CURRENT_TIMESTAMP
            WHERE CAST(hash AS INTEGER) BETWEEN ? AND ?
            """,
            (start, end),
        )
        conn.commit()


def is_already_done(hash_str: str) -> bool:
    """Return True if the given combo hash is already processed (status is not 'pending')."""
    _ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT 1 FROM hashes WHERE hash = ? AND status != 'pending' LIMIT 1",
            (hash_str,),
        )
        return cur.fetchone() is not None


def register_hash(hash_str: str, status: str = "done", error: Optional[str] = None) -> bool:
    """
    Register (or update) a hash in the SQLite registry.

    Args:
        hash_str (str): The combo hash.
        status (str): 'done' (default), 'failed', or custom tag.
        error (Optional[str]): Optional error message if run failed.

    Returns:
        bool: True if newly inserted, False if it already existed.
    """
    _ensure_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        try:
            conn.execute(
                "INSERT INTO hashes (hash, status, error) VALUES (?, ?, ?)",
                (hash_str, status, error),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Entry exists — we can still update its status if desired
            conn.execute(
                "UPDATE hashes SET status = ?, error = ?, timestamp = CURRENT_TIMESTAMP WHERE hash = ?",
                (status, error, hash_str),
            )
            conn.commit()
            return False


def reset_hash_registry(confirm: bool = True) -> None:
    """Safely reset the hash registry."""
    if confirm:
        _ensure_db()
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            conn.execute("DROP TABLE IF EXISTS hashes")
            conn.commit()
        _ensure_db()


def count_hashes(status: Optional[str] = None) -> int:
    """Count total or per-status hashes."""
    _ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        if status:
            cur = conn.execute("SELECT COUNT(*) FROM hashes WHERE status = ?", (status,))
        else:
            cur = conn.execute("SELECT COUNT(*) FROM hashes")
        (count,) = cur.fetchone()
        return count


def get_all_hashes(status: Optional[str] = None) -> List[Tuple[str, str, str, Optional[str]]]:
    """Retrieve all hashes from the SQLite registry, optionally filtered by status."""
    _ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        if status:
            cur.execute("SELECT hash, status, timestamp, error FROM hashes WHERE status = ? ORDER BY timestamp DESC",(status,),)
        else:
            cur.execute("SELECT hash, status, timestamp, error FROM hashes ORDER BY timestamp DESC")
        return cur.fetchall()


def ensure_sequential_hashes(
    max_hash: int,
    *,
    chunk_size: int = 100_000,
) -> None:
    """Ensure the hash registry contains rows for '0'..str(max_hash).

    This function is idempotent: it uses INSERT OR IGNORE so re-running it
    will only add missing rows and leave existing ones untouched.

    Args:
        max_hash: Highest integer hash value to ensure, inclusive. The function
            will ensure that every hash from "0" to str(max_hash) exists.
        chunk_size: Number of rows to batch per executemany() call to avoid
            building a gigantic parameter list in memory.
    """
    _ensure_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        cursor = conn.cursor()
        for start in range(0, max_hash + 1, chunk_size):
            end = min(max_hash + 1, start + chunk_size)
            batch = [(str(i),) for i in range(start, end)]
            cursor.executemany(
                "INSERT OR IGNORE INTO hashes (hash) VALUES (?)",
                batch,
            )
        conn.commit()


def initialize_db(
    total_number_of_tests: int,
    *,
    chunk_size: int = 100_000,
) -> None:
    """Drop and rebuild the hash registry with hashes 0..max_hash.

    Args:
        total_number_of_tests: Highest integer hash value to create, inclusive.
        chunk_size: Batch size for inserts.
    """
    reset_hash_registry(confirm=True)
    ensure_sequential_hashes(max_hash=total_number_of_tests, chunk_size=chunk_size)


def mark_hash_status(
    hash_str: str,
    status: str = "done",
    error: Optional[str] = None,
) -> None:
    """Mark an existing hash row as 'done'.

    This is a lighter-weight helper than `register_hash` for the case where
    the row is guaranteed to exist (e.g., after `ensure_sequential_hashes`).

    Args:
        hash_str: Hash identifier (e.g., "0", "1", "1180676").
        status: Status of the hash row to mark as 'done', 'invalide', etc.
        error: Optional error message to record; defaults to NULL.
    """
    _ensure_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            """
            UPDATE hashes
            SET status = ?,
                error = ?,
                timestamp = CURRENT_TIMESTAMP
            WHERE hash = ?
            """,
            (status, error, hash_str),
        )
        conn.commit()


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


def select_n_imgs(all_items: Iterable[object], n: int = 2, seed: int = 0) -> List[Path]:
    """Randomly select `n` item(s) without replacement.

    Args:
        all_items: List of objects to sample from.
        n: Number of objects to select (1 ≤ n ≤ len(all_items)).
        seed: Seed for reproducibility.

    Returns:
        A list of ``n`` distinct objects.

    Raises:
        ValueError: If ``n`` is outside the valid range.
    """
    total = len(all_items)
    if not (1 <= n <= total):
        raise ValueError(
            f"n must be in [1, {total}], got n={n} with {total} available objects."
        )
    rng = np.random.default_rng(seed)
    return list(rng.choice(all_items, size=n, replace=False))


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


def prepare_images(path_img: Path) -> Dict[str, Any]:
    """
    Prepare images for validation tests.

    Args:
        path_img: Path to the image.

    Returns:
        Dict[str, Any]: Dictionary of image list (ImageListIO).
    """
    out = {"buffers": {0: {0: {}, 1: {0: {}, 1: {}}}, 1: {0: {}, 1: {}}}}
    out['buffers_other'] = copy.deepcopy(out['buffers'])
    out['images'] = None

    rec_standards = [r for r in get_args(REC_STANDARD)]
    images = ImageListIO(input_data=path_img)
    out['images'] = images

    list_buffer = [np.zeros(im.shape, dtype=bool) for im in images]
    for ag in (0, 1):
        for ct in (0, 1):
            for rs in (1, 2, 3):
                # ag, ct, rs = 0, 1, 2

                # Prepare the images
                buffers = ImageListIO(input_data=copy.deepcopy(list_buffer), conserve_memory=False)
                buffers_other = ImageListIO(input_data=copy.deepcopy(list_buffer), conserve_memory=False)
                for idx, image in enumerate(images):
                    buffers[idx] = image.astype(float)
                buffers.drange = (0, 255)

                # Convert them into xyY color space
                rec_stardard = rec_standards[rs - 1]
                output = ColorTreatment.forward_color_treatment(
                    rec_standard=rec_stardard,
                    input_images=buffers,
                    output_images=buffers,
                    output_other=buffers_other,
                    linear_luminance=bool(ct),
                    as_gray=ag,
                )
                _buffers, _buffers_other = output if isinstance(output, tuple) else (output, None)
                out["buffers"][ag][ct][rs] = _buffers
                out["buffers_other"][ag][ct][rs] = _buffers_other
    return out


def precompute_targets(images_buffers: Dict[str, Any]) -> Dict[str, Dict[int, np.ndarray]]:
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
    out = {"hist": {0: {0: {}, 1: {0: {}, 1: {}}}, 1: {0: {}, 1: {}}}, "spec": {0: {0: {}, 1: {}}, 1: {0: {}, 1: {}}}}
    for ag in (0, 1):
        for ct in (0, 1):
            for rs in (1, 2, 3):
                out["hist"][ag][ct][rs] = utils.imhist(images_buffers['buffers'][ag][ct][rs][0])       # (256,)
                out["spec"][ag][ct][rs], _ = utils.image_spectrum(images_buffers['buffers'][ag][ct][rs][0])
    return out
