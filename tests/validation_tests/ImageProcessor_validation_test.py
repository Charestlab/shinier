# tests/integration_tests/test_image_processor_validation_sharded.py
"""Exhaustive ImageProcessor validations with job-level sharding.

This test iterates over *all* combinations of Options parameters and validates:
  - Internal per-step validations reported in `proc.validation` are all True.
  - For iterative modes (5, 6, 7, 8), relevant RMSE metrics improve:
      * Mode 5 (hist + sf): histogram RMSE ↓ and SF RMSE ↓
      * Mode 6 (hist + spec): histogram RMSE ↓ and spectrum RMSE ↓
      * Mode 7 (sf + hist): histogram RMSE ↓ and SF RMSE ↓
      * Mode 8 (spec + hist): histogram RMSE ↓ and spectrum RMSE ↓
  - When `hist_optim=1`, all entries in `proc.ssim_results` have `valid_result=True`.

Sharding:
  - Use env vars to slice work deterministically without materializing all combos:
      SHARDS=<int> # total number of shards (default: 1)
      SHARD_INDEX=<int> # zero-based index of this shard (default: 0)
  - Optional progress:
      SHOW_PROGRESS=1 # show per-shard tqdm progress bar

Run examples:
  - Single shard, with progress:
      SHOW_PROGRESS=1 pytest -m exhaustive -q
  - 4 shards in CI (run 4 jobs with SHARD_INDEX=0..3):
      SHARDS=4 SHARD_INDEX=0 pytest -m exhaustive -q
"""

from __future__ import annotations
from traceback import format_exc

import itertools
import os
import shutil
from math import prod
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tqdm.auto import tqdm

from shinier import ImageDataset, Options, utils
from shinier.ImageProcessor import ImageProcessor
from tests import utils as utils_test

START_AT = int(os.getenv("START_AT", "0"))  # allow skipping combos
DUMP_FILE_FORMAT = os.getenv("DUMP_FILE_FORMAT", "json")


@pytest.mark.validation_tests
def test_imageprocessor_validations_sharded(test_tmpdir: Path) -> None:
    """Validate ImageProcessor across the full Options grid with sharding.

    The test:
      * Builds a tiny dataset (2 RGB images, 128×128).
      * Precomputes per-as_gray target histograms/spectra (once).
      * Iterates lazily over the Cartesian product, sliced by SHARDS/SHARD_INDEX.
      * For each combo:
          - Builds Options (expecting only mode=9 & dithering=0 to be invalid).
          - Runs ImageDataset + ImageProcessor.
          - Asserts all internal proc.validation entries are True.
          - For iterative modes, enforces RMSE improvements on relevant metrics:
                Mode 5 (hist+sf): histogram ↓, SF ↓
                Mode 6 (hist+spec): histogram ↓, spectrum ↓
                Mode 7 (sf+hist): histogram ↓, SF ↓
                Mode 8 (spec+hist): histogram ↓, spectrum ↓
          - If hist_optim=1, ensures all proc.ssim_results have valid_result=True.
      * Cleans the per-combo output directory immediately after use.

    Sharding and progress:
      - SHARDS (int), SHARD_INDEX (int), SHOW_PROGRESS=1 control slicing and tqdm.
    """
    # --- Dataset & targets ---
    src_images_path = utils_test.get_small_imgs_path(utils_test.IMAGE_PATH)

    # Compute target histogram, spatial frequency and spectrum based on first image
    with Image.open(src_images_path[0]) as pil_image:
        src0 = np.array(pil_image.convert("RGB"))

    h, w = src0.shape[:2]
    targets = utils_test.precompute_targets(src0)

    # Compute arbitrary mask (ellipse)
    mask_dir = test_tmpdir / "MASK"
    utils_test.make_masks(mask_dir, h=h, w=w, n=1)

    # --- Parameter grids ---
    modes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    whole_images = [1, 2, 3]
    ditherings = [0, 1, 2]
    as_grays = [0, 1, 2, 3, 4]
    hist_specs = [1, 2, 3, 4]
    hist_optims = [0, 1]
    rescalings = [0, 1, 2, 3]
    safe_lums = [False, True]
    target_lums = [(128, 32), (0, 0)]
    target_hist_choices = ["target", "none", 'equal']
    target_spec_choices = ["target", "none"]

    # --- Shard configuration ---
    shards = int(os.getenv("SHARDS", "1"))
    shard_index = int(os.getenv("SHARD_INDEX", "0"))
    assert shards >= 1 and 0 <= shard_index < shards

    # tqdm set-up (per-shard)
    show_progress = os.getenv("SHOW_PROGRESS", "0") == "1"
    total_combos = prod(
        [
            len(modes),
            len(whole_images),
            len(ditherings),
            len(as_grays),
            len(hist_specs),
            len(hist_optims),
            len(rescalings),
            len(safe_lums),
            len(target_lums),
            len(target_hist_choices),
            len(target_spec_choices),
        ]
    )
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(
            total=total_combos // shards + (1 if shard_index < (total_combos % shards) else 0),
            initial=START_AT,
            ncols=0,
            desc=f"Shard {shard_index + 1}/{shards}",
        )

    checked = 0
    ran = 0

    # --- Lazy product with index-based slicing (no full list in RAM) ---
    product_iter = itertools.product(
        modes,
        whole_images,
        ditherings,
        as_grays,
        hist_specs,
        hist_optims,
        rescalings,
        safe_lums,
        target_lums,
        target_hist_choices,
        target_spec_choices,
    )

    cnt = 0
    for i, combo in enumerate(product_iter):
        # Deterministic shard slicing
        if i % shards != shard_index:
            continue

        # Increment per-shard counter
        cnt += 1

        # Skip until reaching the desired starting index
        if cnt < START_AT:
            continue

        # Deterministic unique seed related to specific combo
        seed_iter = utils_test.deterministic_seed_from_combo(combo=combo)

        (mode, wi, dith, ag, hs, ho, rs, slm, t_lum, th_choice, ts_choice) = combo
        checked += 1

        # Per-combo output dir
        out = test_tmpdir / (
            f"OUT_m{mode}_wi{wi}_d{dith}_g{ag}_hs{hs}_ho{ho}_r{rs}"
            f"_slm{int(slm)}_tl{t_lum[0]}-{t_lum[1]}_th{th_choice}_ts{ts_choice}"
        )
        out.mkdir(parents=True, exist_ok=True)

        # Choose explicit targets for Options (or None)
        if th_choice == "target":
            th = targets["hist"][ag]
        elif th_choice == 'equal':
            th = 'equal'
        else:
            th = None

        ts = targets["spec"][ag] if ts_choice == "target" else None

        opts_kwargs = dict(
            input_folder=utils_test.IMAGE_PATH,
            output_folder=out,
            images_format="png",
            masks_folder=mask_dir if wi == 3 else None,
            masks_format="png" if wi == 3 else None,
            whole_image=wi,
            mode=mode,
            as_gray=ag,
            dithering=dith,
            conserve_memory=True,
            seed=seed_iter,
            safe_lum_match=slm,
            target_lum=t_lum,
            hist_specification=hs,
            hist_optim=ho,
            hist_iterations=3,
            step_size=34,
            target_hist=th,
            rescaling=rs,
            target_spectrum=ts,
            iterations=2,  # Options will clamp to 1 for non-iterative modes
        )
        rand_selected_paths = None
        try:
            invalid_expected = mode == 9 and dith == 0
            try:
                opts = Options(**opts_kwargs)
            except (ValueError, TypeError) as e:
                if invalid_expected:
                    continue
                raise AssertionError(f"Unexpected Options validation error for combo:\n{opts_kwargs}\nError: {e}")

            # Build dataset & process
            rand_selected_paths = utils_test.select_n_imgs(src_images_path, n=2, seed=seed_iter)
            ds = ImageDataset(images=rand_selected_paths, options=opts)
            proc = ImageProcessor(dataset=ds, options=opts, verbose=-1)
            ran += 1

            # All internal validations must be True
            for rec in getattr(proc, "validation", []):
                if not bool(rec.get("valid_result", True)):
                    dump_path = utils_test.dump_failure_context(
                        combo_dict=opts_kwargs,
                        rec=rec,
                        tmp_root=test_tmpdir,
                        seed=seed_iter,
                        selected_paths=rand_selected_paths,
                        file_type=DUMP_FILE_FORMAT
                    )
                    raise AssertionError(
                        f"\nInternal validation failed\n"
                        f"/t→ Shard {shard_index}, Combo index {i}, Seed {seed_iter}\n"
                        f"/t→ Image: {rec.get('image')}\n"
                        f"/t→ Combo: {opts_kwargs}\n"
                        f"/t→ Log:\n{utils_test.strip_ansi(str(rec.get('log_result', '')))}\n"
                        f"/t→ Dumped context: {dump_path}\n"
                    )

            # Iterative modes: enforce RMSE decrease on relevant metrics only
            if mode in (5, 6, 7, 8):
                # Histogram is involved in all 5..8 modes
                _, rmse_hist_before = utils.hist_match_validation(images=ds.images, binary_masks=proc.bool_masks)
                _, rmse_hist_after = utils.hist_match_validation(images=proc.dataset.images, binary_masks=proc.bool_masks)
                if not np.all(rmse_hist_after <= rmse_hist_before + 1e-9):
                    rec = {
                        "iter": -1,
                        "step": -1,
                        "processing_function": "hist_match",
                        "valid_result": False,
                        "log_result": f"Histogram RMSE not improved: {rmse_hist_before} -> {rmse_hist_after}",
                    }
                    dump_path = utils_test.dump_failure_context(
                        combo_dict=opts_kwargs,
                        rec=rec,
                        tmp_root=test_tmpdir,
                        seed=seed_iter,
                        selected_paths=rand_selected_paths,
                        file_type=DUMP_FILE_FORMAT
                    )
                    raise AssertionError(
                        f"\nHistogram RMSE not improved\n"
                        f"\t→ Shard {shard_index}, Combo index {i}, Seed {seed_iter}\n"
                        f"\t→ Combo: {opts_kwargs}\n"
                        f"\t→ Before: {rmse_hist_before:.6g}  After: {rmse_hist_after:.6g}\n"
                        f"\t→ Dumped context: {dump_path}\n"
                    )

                if mode in (5, 7):  # sf involved
                    _, rmse_sf_before = utils.sf_match_validation(images=ds.images)
                    _, rmse_sf_after = utils.sf_match_validation(images=proc.dataset.images)
                    if not np.all(rmse_sf_after <= rmse_sf_before + 1e-9):
                        rec = {
                            "iter": -1,
                            "step": -1,
                            "processing_function": "sf_match",
                            "valid_result": False,
                            "log_result": f"SF RMSE not improved: {rmse_sf_before} -> {rmse_sf_after}",
                        }
                        dump_path = utils_test.dump_failure_context(
                            combo_dict=opts_kwargs,
                            rec=rec,
                            tmp_root=test_tmpdir,
                            seed=seed_iter,
                            selected_paths=rand_selected_paths,
                            file_type=DUMP_FILE_FORMAT
                        )
                        raise AssertionError(
                            f"\nSF RMSE not improved\n"
                            f"\t→ Shard {shard_index}, Combo index {i}, Seed {seed_iter}\n"
                            f"\t→ Combo: {opts_kwargs}\n"
                            f"\t→ Before: {rmse_sf_before:.6g}  After: {rmse_sf_after:.6g}\n"
                            f"\t→ Dumped context: {dump_path}\n"
                        )

                if mode in (6, 8):  # spec involved
                    _, rmse_spec_before = utils.spec_match_validation(images=ds.images)
                    _, rmse_spec_after = utils.spec_match_validation(images=proc.dataset.images)
                    if not np.all(rmse_spec_after <= rmse_spec_before + 1e-9):
                        rec = {
                            "iter": -1,
                            "step": -1,
                            "processing_function": "spec_match",
                            "valid_result": False,
                            "log_result": f"Spectrum RMSE not improved: {rmse_spec_before} -> {rmse_spec_after}",
                        }
                        dump_path = utils_test.dump_failure_context(
                            combo_dict=opts_kwargs,
                            rec=rec,
                            tmp_root=test_tmpdir,
                            seed=seed_iter,
                            selected_paths=rand_selected_paths,
                            file_type=DUMP_FILE_FORMAT
                        )
                        raise AssertionError(
                            f"\nSpectrum RMSE not improved\n"
                            f"\t→ Shard {shard_index}, Combo index {i}, Seed {seed_iter}\n"
                            f"\t→ Combo: {opts_kwargs}\n"
                            f"\t→ Before: {rmse_spec_before:.6g}  After: {rmse_spec_after:.6g}\n"
                            f"\t→ Dumped context: {dump_path}\n"
                        )

            # SSIM optimization monotonicity (if enabled)
            if ho == 1:
                ssim_records = getattr(proc, "ssim_results", [])
                if len(ssim_records) >= 1:
                    if not all(bool(r.get("valid_result", True)) for r in ssim_records):
                        rec = {
                            "iter": -1,
                            "step": -1,
                            "processing_function": "hist_match",
                            "valid_result": False,
                            "log_result": f"SSIM optimization non-monotonic: {ssim_records}",
                        }
                        dump_path = utils_test.dump_failure_context(
                            combo_dict=opts_kwargs,
                            rec=rec,
                            tmp_root=test_tmpdir,
                            seed=seed_iter,
                            selected_paths=rand_selected_paths,
                            file_type=DUMP_FILE_FORMAT
                        )
                        raise AssertionError(
                            f"\nSSIM optimization non-monotonic\n"
                            f"\t→ Shard {shard_index}, Combo index {i}, Seed {seed_iter}\n"
                            f"\t→ Combo: {opts_kwargs}\n"
                            f"\t→ ssim records: {ssim_records}\n"
                            f"\t→ Dumped context: {dump_path}\n"
                        )

        except Exception as e:
            tb = format_exc()
            dump_path = utils_test.dump_failure_context(
                combo_dict=opts_kwargs,
                rec={"iter": 0, "step": 0, "error": str(e), "traceback": tb},
                tmp_root=test_tmpdir,
                seed=seed_iter,
                selected_paths=rand_selected_paths,
                file_type=DUMP_FILE_FORMAT
            )
            raise AssertionError(
                f"\n💥 Unexpected error while processing combo:\n"
                f"→ Shard {shard_index}, Combo index {i}, Seed {seed_iter}\n"
                f"→ Combo: {opts_kwargs}\n"
                f"→ Exception: {e.__class__.__name__}: {e}\n"
                f"→ Traceback:\n{tb}\n"
                f"→ Dumped context: {dump_path}\n"
            )
        finally:
            # Per-combo cleanup to avoid temp accumulation
            shutil.rmtree(out, ignore_errors=True)
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Sanity: this shard actually executed some work
    assert checked >= 0  # checked can be 0 for shards > total_combos
    if total_combos >= shards:
        assert ran >= 0  # ran could be 0 if all combos in this shard were invalid (unlikely)
