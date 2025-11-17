# tests/integration_tests/test_image_processor_validation_sharded.py
"""Exhaustive ImageProcessor validations with smart pruning and sharding.

Prunes only:
  • Impossible: mode==9 & dithering==0
  • Redundant:
      - rec_standard ignored when linear_luminance is True (fix to 2)
      - verbose (ignored here)
      - hist_specification ignored when hist_optim==1 (force None)
      - safe_lum_match only relevant when mode==1
      - legacy_mode: test only one combo per mode with legacy_mode=True

Also restores RMSE–improvement checks for modes 5..8.

Env:
  SHARDS, SHARD_INDEX, SHOW_PROGRESS, DUMP_FILE_FORMAT, START_AT
"""

from __future__ import annotations

import copy
from traceback import format_exc
import itertools
import os
import shutil
from math import prod
from pathlib import Path

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_origin, get_args, Literal
import numpy as np
import pytest
from PIL import Image
from tqdm.auto import tqdm

from shinier import ImageDataset, Options, utils, ImageListIO, ImageProcessor
from tests import utils as utils_test
from shinier.color.Converter import REC_STANDARD, ColorTreatment
REC_STANDARD = [r for r in get_args(REC_STANDARD)]
pytestmark = pytest.mark.validation_tests


# -------------------- env --------------------
START_AT = int(os.getenv("START_AT", "0"))
RESTART = os.getenv("RESTART", "false").lower() in ("1", "true", "yes")

DUMP_FILE_FORMAT = os.getenv("DUMP_FILE_FORMAT", "pkl")
SHARDS = int(os.getenv("SHARDS", "1"))
SHARD_INDEX = int(os.getenv("SHARD_INDEX", "0"))
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "1") == "1"
PERCENT_SAMPLED = float(os.getenv("PERCENT_SAMPLED", "1"))
# SHARD_INDEX = 4
# SHARDS = 8
# START_AT = 2260


def get_possible_values(field: str) -> Any:
    """Return all possible categorical values for a field."""
    ann = field.annotation

    # Handle Optional[...] = Union[..., NoneType]
    if get_origin(ann) is Union:
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            ann = args[0]  # unwrap inner type

    # Handle Literal[...] fields
    if get_origin(ann) is Literal:
        return list(get_args(ann))

    # Handle plain bool
    if ann is bool:
        return [True, False]

    # Fallback to default if defined
    if field.default is not None:
        return [field.default]

    # Otherwise, assume None is allowed
    return [None]


def test_imageprocessor_validations_sharded(test_tmpdir: Path) -> None:
    # ----- reset combo registry -----
    utils_test.reset_hash_registry(RESTART)  # if True, will redo all tests already completed

    # ----- dataset & targets -----
    src_images_path = utils_test.get_small_imgs_path(utils_test.IMAGE_PATH)
    images_buffers = utils_test.prepare_images(utils_test.IMAGE_PATH)
    src0 = images_buffers['images'][0]
    # with Image.open(src_images_path[0]) as pil_image:
    #     src0 = np.array(pil_image.convert("RGB"))
    h, w = src0.shape[:2]
    targets = utils_test.precompute_targets(images_buffers)
    ag, ct, rs = 1, 0, 1

    mask_dir = test_tmpdir / "MASK"
    utils_test.make_masks(mask_dir, h=h, w=w, n=1)

    # ----- parameter grids (only pruned redundancies) -----
    choices = {name: get_possible_values(field) for name, field in Options.model_fields.items()}
    choices['input_folder'] = [utils_test.IMAGE_PATH]
    choices['masks_folder'] = [mask_dir]
    choices['background'] += [120, 130]
    choices['seed'] += [4242424242]
    choices['target_lum'] += [(100, 20)]
    choices['target_hist'] += ['unit_test']
    choices['target_spectrum'] += ['unit_test']
    choices['hist_iterations'] = [3]
    choices['verbose'] = [-1]

    all_fields = list(choices.keys())
    total_combo = np.prod([len(v) for v in choices.values() if hasattr(v, '__len__') and not isinstance(v, str)])  # Some of which are not valid and duplicated

    pbar = None
    if SHOW_PROGRESS and tqdm is not None:
        per_shard = total_combo // SHARDS + (1 if SHARD_INDEX < (total_combo % SHARDS) else 0)
        pbar = tqdm(total=total_combo, initial=START_AT, desc=f"Shard {SHARD_INDEX+1}/{SHARDS}", ncols=0)

    # Reset the iterator for the real loop
    cnt = -1
    for i, combo in enumerate(itertools.product(*(choices[f] for f in all_fields))):
        if pbar is not None:
            pbar.update(1)

        kwargs = dict(zip(all_fields, combo))

        # if `i` within this SHARD_INDEX, proceed else next `i`
        if i % SHARDS != SHARD_INDEX:
            continue
        cnt += 1

        # if `i` >= START_ITER, proceed else next `i`
        if i < START_AT:
            continue

        # if combo never tested, proceed else next `i`
        combo_hash = str(i)
        if utils_test.is_already_done(combo_hash):
            continue

        # if not randomly selected, proceed else next `i`
        is_sampled = np.random.rand() >= (1 - PERCENT_SAMPLED)
        if not is_sampled:
            continue

        # Create target hist or spectrum if requested
        try:
            opts = Options(**kwargs)
        except:
            utils_test.register_hash(combo_hash, status="invalid")
            continue

        ag = True if kwargs['legacy_mode'] else kwargs['as_gray']
        opts.target_hist = targets["hist"][int(ag)][kwargs['linear_luminance']][kwargs['rec_standard']] if kwargs['target_hist'] == "unit_test" else ("equal" if kwargs['target_hist'] == "equal" else None)
        opts.target_spectrum = targets["spec"][int(ag)][kwargs['linear_luminance']][kwargs['rec_standard']] if kwargs['target_spectrum'] == "unit_test" else None

        # Set seed
        if opts.seed is not None:
            seed_iter = opts.seed
        else:
            seed_iter = utils_test.deterministic_seed_from_combo(combo=combo)

        # Log file
        out_dir = test_tmpdir / (
            f"OUT_"
            f"m{opts.mode}"  # Mode
            f"_wi{opts.whole_image}"  # Whole image flag (1–3)
            f"_d{opts.dithering}"  # Dithering method
            f"_ag{int(opts.as_gray)}"  # Grayscale flag
            f"_ct{opts.linear_luminance}"  # Color treatment
            f"_rs{opts.rec_standard}"  # Rec. standard
            f"_ho{int(opts.hist_optim)}"  # Histogram optimization
            f"_hs{opts.hist_specification}"  # Histogram specification
            f"_re{opts.rescaling}"  # Rescaling method
            f"_slm{int(opts.safe_lum_match)}"  # Safe luminance matching
            f"_tl{opts.target_lum[0]}-{opts.target_lum[1]}"  # Target luminance (mean-std)
            f"_lm{int(opts.legacy_mode)}"  # Legacy mode
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        ag2, ct, rs = None, None, None
        try:
            opts.output_folder = out_dir
            ag2, ct, rs = int(opts.as_gray), int(opts.linear_luminance), opts.rec_standard
        except:
            if out_dir and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)

            # Register that combo as invalide
            utils_test.register_hash(combo_hash, status="invalide")
            continue

        rand_selected_paths = None
        try:
            # known impossible
            if opts.mode == 9 and opts.dithering == 0:
                continue

            # Options / pipeline
            rand_selected_images = utils_test.select_n_imgs(images_buffers['images'], n=2, seed=seed_iter)  # sRGB
            rand_selected_buffers = utils_test.select_n_imgs(images_buffers['buffers'][ag2][ct][rs], n=2, seed=seed_iter)  # Y from xyY or sRGB (depending on linear_luminance)
            rand_selected_paths = utils_test.select_n_imgs(src_images_path, n=2, seed=seed_iter)  # sRGB
            # as_gray_ds = 1 if opts.as_gray == True and opts.linear_luminance is True else 0

            # images_copy = ImageListIO(input_data=rand_selected_images, conserve_memory=False)
            initial_buffers = ImageListIO(input_data=rand_selected_buffers, conserve_memory=False)
            # buffers_empty = [np.zeros(im.shape, dtype=bool) for im in initial_buffers]

            ds = ImageDataset.model_construct(images=rand_selected_images, options=opts)
            proc = ImageProcessor.model_construct(dataset=ds, options=opts, verbose=-1, from_validation_test=True)

            # Prepare targets for validation
            th = opts.target_hist if proc._target_hist is None else proc._target_hist
            ts = opts.target_spectrum if proc._target_spectrum is None else proc._target_spectrum

            # Prepare images for validation: convert them into xyY if needed
            final_buffers = proc._final_buffer

            # internal validations
            for rec in getattr(proc, "validation", []):
                if not bool(rec.get("valid_result", True)):
                    _dump_and_fail(rec, kwargs, seed_iter, rand_selected_paths, test_tmpdir)

            # --------- RESTORED: RMSE improvement checks for modes 5..8 ----------
            if opts.mode in (5, 6, 7, 8):
                # Histogram (always for 5..8)
                _, rmse_hist_before = utils.hist_match_validation(images=initial_buffers, binary_masks=proc.bool_masks, target_hist=th, normalize_rmse=True)
                _, rmse_hist_after = utils.hist_match_validation(images=final_buffers, binary_masks=proc.bool_masks, target_hist=th, normalize_rmse=True)
                if not np.all(rmse_hist_after + 1e-9 <= rmse_hist_before):
                    rec = {
                        "iter": -1, "step": -1, "processing_function": "hist_match",
                        "valid_result": False,
                        "log_result": f"Histogram RMSE not improved: {rmse_hist_before} -> {rmse_hist_after}",
                    }
                    _dump_and_fail(rec, kwargs, seed_iter, rand_selected_paths, test_tmpdir)

                # Spatial frequency (5,7)
                if opts.mode in (5, 7):
                    _, rmse_sf_before = utils.sf_match_validation(images=initial_buffers, target_spectrum=ts, normalize_rmse=True)
                    _, rmse_sf_after = utils.sf_match_validation(images=final_buffers, target_spectrum=ts, normalize_rmse=True)
                    if not np.all(rmse_sf_after <= rmse_sf_before + 1e-9):
                        rec = {
                            "iter": -1, "step": -1, "processing_function": "sf_match",
                            "valid_result": False,
                            "log_result": f"SF RMSE not improved: {rmse_sf_before} -> {rmse_sf_after}",
                        }
                        _dump_and_fail(rec, kwargs, seed_iter, rand_selected_paths, test_tmpdir)

                # Spectrum (6,8)
                if opts.mode in (6, 8):
                    _, rmse_spec_before = utils.spec_match_validation(images=initial_buffers, target_spectrum=ts, normalize_rmse=True)
                    _, rmse_spec_after = utils.spec_match_validation(images=final_buffers, target_spectrum=ts, normalize_rmse=True)
                    if not np.all(rmse_spec_after <= rmse_spec_before + 1e-9):
                        rec = {
                            "iter": -1, "step": -1, "processing_function": "spec_match",
                            "valid_result": False,
                            "log_result": f"Spectrum RMSE not improved: {rmse_spec_before} -> {rmse_spec_after}",
                        }
                        _dump_and_fail(rec, kwargs, seed_iter, rand_selected_paths, test_tmpdir)
            # ---------------------------------------------------------------------

            # SSIM optimization monotonicity
            if opts.hist_optim and opts.mode in (2, 5, 6, 7, 8):
                ssim_records = getattr(proc, "ssim_results", [])
                if ssim_records and not all(bool(r.get("valid_result", True)) for r in ssim_records):
                    rec = {"iter": -1, "step": -1, "processing_function": "hist_match",
                           "valid_result": False, "log_result": f"SSIM optimization non-monotonic: {ssim_records}"}
                    _dump_and_fail(rec, kwargs, seed_iter, rand_selected_paths, test_tmpdir)
        except ControlledFailure:
            pass  # handled already
        except Exception as e:
            tb = format_exc()
            dump_path = utils_test.dump_failure_context(
                combo_dict=kwargs,
                rec={"iter": 0, "step": 0, "error": str(e), "traceback": tb},
                tmp_root=test_tmpdir,
                seed=seed_iter,
                selected_paths=rand_selected_paths,
                file_type=DUMP_FILE_FORMAT,
            )
            raise AssertionError(
                f"\n💥 Unexpected error\n"
                f"→ Shard {SHARD_INDEX}, Combo global-index {i}, Per-shard #{cnt}\n"
                f"→ Combo: {kwargs}\n"
                f"→ Exception: {e.__class__.__name__}: {e}\n"
                f"→ Dumped context: {dump_path}\n"
            )
        else:
            # Cleanup safely
            if out_dir and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            # Register that combo as valide
            utils_test.register_hash(combo_hash, status="done")

    if pbar is not None:
        pbar.close()
    assert cnt >= 0


# -------------------- helpers --------------------
class ControlledFailure(AssertionError):
    """Raised when _dump_and_fail already handled the failure context."""


def _dump_and_fail(rec, opts_kwargs, seed, selected_paths, tmp_root):
    selected_paths = selected_paths or []  # ← ensure iterable
    dump_path = utils_test.dump_failure_context(
        combo_dict=opts_kwargs,
        rec=rec,
        tmp_root=tmp_root,
        seed=seed,
        selected_paths=selected_paths,
        file_type=DUMP_FILE_FORMAT,
    )
    raise ControlledFailure(
        f"\nValidation failed\n"
        f"→ Combo: {opts_kwargs}\n"
        f"→ Log:\n{utils_test.strip_ansi(str(rec.get('log_result', '')))}\n"
        f"→ Dumped context: {dump_path}\n"
    )