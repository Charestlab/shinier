# tests/tools/replay_failure.py
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from pprint import pprint
from PIL import Image
import numpy as np

from tests import utils  # your helper utilities
from shinier import Options, ImageDataset
from shinier import utils as util
from shinier.ImageProcessor import ImageProcessor

# ---------------------------------------------------------------------------
# Allow imports from repo root (so helpers & src work outside pytest)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _default_base_tmp() -> Path:
    """Return the unified base tmp directory for replays/tests.

    The default is `<repo>/tests/IMAGES/tmp`, but this can be overridden
    by setting the `SHINIER_BASE_TMP` environment variable.

    Returns:
        Path: Absolute path to the base temporary directory.
    """
    default_base = Path(__file__).resolve().parents[2] / "tests" / "IMAGES" / "tmp"
    return Path(os.getenv("SHINIER_BASE_TMP", default_base)).resolve()


def main(file_path: Path, src_path: Optional[str] = None) -> None:
    """Replay a failure context for debugging.

    This loads a JSON file dumped by the exhaustive validation tests,
    reconstructs the `Options`, reuses the selected images, coerces them
    to RGB if needed, then runs `ImageProcessor` so you can debug with
    breakpoints.

    Args:
        file_path: Path to the failure_<id> file dumped during exhaustive tests.

    Raises:
        FileNotFoundError: If the provided json_path does not exist.
        KeyError: If expected keys are missing from the JSON payload.
        ValueError: If any value conversion fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Failure context file not found: {file_path}")

    # --- Load context ---
    if file_path.suffix not in ['.pkl', '.json']:
        raise ValueError('Should be a pkl or json file.')
    if file_path.suffix == '.pkl':
        ctx = utils.load_pickle(file_path)
    else:
        ctx = utils.load_json(file_path)

    # Required keys (will raise KeyError if missing; that's intentional)
    combo = ctx["combo_opts_kwargs"]
    seed = ctx["seed"]
    selected_images = [Path(p) for p in ctx["selected_images"]]
    record = ctx["record"]

    print("\n=== Replaying failure ===")
    pprint(combo)
    print(f"Seed: {seed}")
    print(f"Selected images: {selected_images}")
    print(f"Record:\n{record}")

    # --- Rebuild combo types ---
    # Convert back stringified paths and seed
    out_dir = Path(combo["output_folder"])
    combo["output_folder"] = out_dir

    if combo.get("input_folder") and combo["input_folder"] != "None":
        combo["input_folder"] = Path(combo["input_folder"])
    else:
        combo["input_folder"] = None

    combo["seed"] = int(seed)
    # combo['mode'] = 7
    # combo['target_spectrum'] = None

    # Ensure output folder exists (so downstream saves don't explode)
    out_dir.mkdir(parents=True, exist_ok=True)
    # src_path = combo['input_folder']
    src_images_path = utils.get_small_imgs_path(src_path)
    with Image.open(src_images_path[0]) as pil_image:
        src0 = np.array(pil_image.convert("RGB"))
    h, w = src0.shape[:2]
    targets = utils.precompute_targets(src0)

    # # Make mask
    # mask_path_name = combo['output_folder'] / 'mask'
    # utils.make_masks(mask_path_name, h=h, w=w, n=2)
    # combo['masks_folder'] = mask_path_name
    # combo['whole_image'] = 3
    # combo['mode'] = 5
    # combo['masks_format'] = 'png'
    combo['verbose'] = 2

    # --- Build Options ---
    opts = Options(**combo)
    # opts.target_hist = targets['hist'][0]
    # opts.target_hist = None

    # --- Build Dataset ---
    # Ensure images are RGB to avoid target_hist shape issues
    # (write coerced copies alongside output dir to avoid polluting sources)
    coerced_dir = out_dir / "replay_rgb"
    coerced_imgs = utils.coerce_to_rgb(selected_images, coerced_dir)
    ds = ImageDataset(images=coerced_imgs, options=opts)

    # --- Run Processor ---
    proc = ImageProcessor(dataset=ds, options=opts)

    # Histogram is involved in all 5..8 modes
    _, rmse_hist_before = util.hist_match_validation(images=ds.images)
    _, rmse_hist_after = util.hist_match_validation(images=proc.dataset.images)
    _, rmse_sf_before = util.sf_match_validation(images=ds.images)
    _, rmse_sf_after = util.sf_match_validation(images=proc.dataset.images)

    print("\nReplay completed.")
    print("You can now set breakpoints inside ImageProcessor and re-run for debugging.")


if __name__ == "__main__":
    try:
        file_path = Path(sys.argv[1])
        main(file_path)
    finally:
        src_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/SAMPLE_64X64')
        # src_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/SAMPLE_512X512')
        # src_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/SAMPLE_1024X124')

        # file_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/tmp/shard0-of-8/master/case-42ec98b3eaf2/failure_05410ad6.pkl')
        # file_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/tmp/shard0-of-8/master/case-609dab2fd6e2/failure_5d1f75f4.pkl')
        file_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/tmp/shard0-of-8/master/case-d73efbdf68da/failure_dcfd264c.pkl')
        # file_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/tmp/shard0-of-8/master/case-bb6ed7c15ec8/failure_b866525e.pkl')
        # file_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/tmp/shard0-of-8/master/case-98e0c29c9110/failure_df5942e4.json')
        # file_path = Path('/Users/ndr/GIT_REPO/GITHUB/shine/shinier/tests/IMAGES/tmp/shard3-of-8/master/case-8b448e852111/failure_dbd8c896.json')
        main(file_path, src_path)

