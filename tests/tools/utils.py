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
import numpy as np
from PIL import Image
from shinier import utils
from shinier.ImageListIO import ImageListIO
from shinier.color import ColorTreatment, REC_STANDARD, RGB_STANDARD

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Mapping for utils.rgb2gray's weighting_standard (when precomputing targets)
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
    Tuple[Optional[float], Optional[float]],  # target_lum
    str,  # target_hist_choice ("target" | "none")
    str,  # target_spec_choice ("target" | "none")
]

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
DB_PATH = Path(__file__).resolve().parent.parent / "hash_registry.db"

# Only terminal-success statuses are skipped on a re-run. 'failed'/'error' combos
# are intentionally NOT skipped so a real regression stays red until it is fixed.
TERMINAL_SKIP_STATUSES = ("done", "invalid", "invalid_option_combination")


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


def mark_hash_range_done(start: int, end: int, namespace: str = "", chunk_size: int = 100_000) -> None:
    """Upsert all integer hashes in [start, end] as 'done' (for START_AT skipping).
    Args:
        start: Lowest integer hash value (inclusive).
        end: Highest integer hash value (inclusive).
        namespace: Prefix used when building hash keys (e.g. COVERAGE_MODE).
        chunk_size: Rows per executemany batch.
    """
    _ensure_db()
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")
    prefix = f"{namespace}:" if namespace else ""
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        for batch_start in range(start, end + 1, chunk_size):
            batch_end = min(end + 1, batch_start + chunk_size)
            batch = [
                (f"{prefix}{i}", 'done')
                for i in range(batch_start, batch_end)
            ]
            conn.executemany(
                """
                INSERT INTO hashes (hash, status) VALUES (?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    status = excluded.status,
                    timestamp = CURRENT_TIMESTAMP
                """,
                batch,
            )
        conn.commit()


def is_already_done(hash_str: str) -> bool:
    """Return True only if the combo reached a terminal-success status.

    Combos marked 'failed' or 'error' are deliberately reported as *not* done so
    that re-running the suite retries them; otherwise a single recorded failure
    would be skipped forever and the test would go green without a fix.
    """
    _ensure_db()
    placeholders = ",".join("?" * len(TERMINAL_SKIP_STATUSES))
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            f"SELECT 1 FROM hashes WHERE hash = ? AND status IN ({placeholders}) LIMIT 1",
            (hash_str, *TERMINAL_SKIP_STATUSES),
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



def initialize_db() -> None:
    """Drop and recreate the hash registry (empty — rows are upserted on demand)."""
    reset_hash_registry(confirm=True)


def mark_hash_status(
    hash_str: str,
    status: str = "done",
    error: Optional[str] = None,
) -> None:
    """Upsert a hash row with the given status.

    Args:
        hash_str: Hash identifier (e.g., "pruned:1180676").
        status: Status to record ('done', 'failed', 'invalid', etc.).
        error: Optional error message to record; defaults to NULL.
    """
    _ensure_db()
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute(
            """
            INSERT INTO hashes (hash, status, error) VALUES (?, ?, ?)
            ON CONFLICT(hash) DO UPDATE SET
                status = excluded.status,
                error = excluded.error,
                timestamp = CURRENT_TIMESTAMP
            """,
            (hash_str, status, error),
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
    out['images'] = None

    rec_standards = [r for r in get_args(REC_STANDARD)]
    images = ImageListIO(input_data=path_img)
    out['images'] = images

    # Memory: only the first image's color-treated buffer is ever consumed downstream
    # (precompute_targets reads [...][0]). Building the full color-treated set for all 12
    # (as_gray, linear_luminance, rec) cells — plus the never-read buffers_other — would
    # hold ~24 full-dataset copies (O(n_images * H * W), prohibitive at large resolutions).
    # Treat a single-image collection per cell instead, so the footprint stays ~O(1 image).
    first_mask = np.zeros(images[0].shape, dtype=bool)
    for ag in (0, 1):
        for ct in (0, 1):
            for rs in (1, 2, 3):
                buffers = ImageListIO(input_data=[first_mask.copy()], conserve_memory=False)
                buffers[0] = images[0].astype(float)
                buffers.drange = (0, 255)
                # forward_color_treatment requires a sink for chroma channels on the
                # color-preserving paths; we provide a throwaway and discard it.
                other = ImageListIO(input_data=[first_mask.copy()], conserve_memory=False)
                output = ColorTreatment.forward_color_treatment(
                    rec_standard=rec_standards[rs - 1], input_images=buffers, output_images=buffers,
                    linear_luminance=bool(ct), as_gray=ag, output_other=other)
                _buffers = output[0] if isinstance(output, tuple) else output
                out["buffers"][ag][ct][rs] = _buffers
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
