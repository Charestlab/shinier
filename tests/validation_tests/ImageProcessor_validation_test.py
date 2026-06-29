# tests/validation_tests/ImageProcessor_validation_test.py
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
  COVERAGE_MODE, SHARDS, SHARD_INDEX, SHOW_PROGRESS, DUMP_FILE_FORMAT,
  START_AT, PERCENT_SAMPLED (sampled mode only). See tests/README.md.
"""

from __future__ import annotations

from traceback import format_exc
import os
import shutil
import random
import hashlib
import json
from pathlib import Path

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_origin, get_args, Literal
import numpy as np
import pytest
from pydantic import ValidationError
from tqdm.auto import tqdm

from shinier import ImageDataset, Options, utils, ImageProcessor
from tests import utils as utils_test
from shinier.color.Converter import REC_STANDARD
REC_STANDARD = [r for r in get_args(REC_STANDARD)]
pytestmark = pytest.mark.validation_tests

# Exceptions Options() raises for *intentionally* invalid combos. Anything outside
# this set is treated as a real bug rather than silently classified as 'invalid'.
EXPECTED_INVALID = (ValidationError, ValueError, TypeError)


# -------------------- env --------------------
START_AT = int(os.getenv("START_AT", "0"))
RESTART = os.getenv("RESTART", "false").lower() in ("1", "true", "yes")

DUMP_FILE_FORMAT = os.getenv("DUMP_FILE_FORMAT", "pkl")
SHARDS = int(os.getenv("SHARDS", "1"))
SHARD_INDEX = int(os.getenv("SHARD_INDEX", "0"))
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "1") == "1"
PERCENT_SAMPLED = float(os.getenv("PERCENT_SAMPLED", "0.001"))
COVERAGE_MODE = os.getenv("COVERAGE_MODE", "sampled")  # exhaustive | pruned | sampled
# Promote soft (quality-regression) anomalies to hard failures when set.
STRICT_SOFT = os.getenv("STRICT_SOFT", "0").lower() in ("1", "true", "yes")
# Aggregate gate: max fraction of executed combos allowed to regress (1.0 = off).
MAX_SOFT_FAIL_RATE = float(os.getenv("MAX_SOFT_FAIL_RATE", "1.0"))

# SHARD_INDEX = 4
# SHARDS = 8
# START_AT = 1180676


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
    # ----- dataset & targets -----
    src_images_path = utils_test.get_small_imgs_path(utils_test.IMAGE_PATH)
    images_buffers = utils_test.prepare_images(utils_test.IMAGE_PATH)
    src0 = images_buffers['images'][0]
    h, w = src0.shape[:2]
    targets = utils_test.precompute_targets(images_buffers)

    mask_dir = test_tmpdir / "MASK"
    utils_test.make_masks(mask_dir, h=h, w=w, n=1)

    # ----- parameter grids -----
    choices = {name: get_possible_values(field) for name, field in Options.model_fields.items()}
    choices['input_folder'] = [utils_test.IMAGE_PATH]
    choices['masks_folder'] = [mask_dir]
    choices['background'] += [120, 130]
    choices['seed'] += [4242424242]
    choices['target_lum'] += [(100, 20), (None, 20), (100, None)]
    choices['target_hist'] += ['unit_test']
    choices['target_spectrum'] += ['unit_test']
    choices['hist_iterations'] = [3]
    choices['verbose'] = [-1]

    # Pruned grid: collapses options whose values share the same code path.
    # Collapsed values are individually verified in PrunedCoverage_test.py.
    choices_pruned = dict(choices)
    choices_pruned['fft_padding_mode'] = [0, 1]       # off vs. on; modes 2/3 are non-distinct
    choices_pruned['standalone_op'] = ['ie_methods']   # dithering path covered by dithering param above
    choices_pruned['ie_methods'] = ['tidhe']           # all methods share the same call site
    choices_pruned['rec_standard'] = [1]               # same code path, different matrices only
    choices_pruned['seed'] = [None]                    # only affects tie-breaking randomness
    choices_pruned['dithering'] = [0, 1]               # off + one ordered method
    choices_pruned['hist_specification'] = [1, 3]      # basic + one advanced variant
    choices_pruned['rescaling'] = [0, 2]               # off + default; variants 1/3 non-distinct
    choices_pruned['conserve_memory'] = [False]        # no code-path interaction with other params
    choices_pruned['background'] = [300, 120]          # 300 = automatic (special sentinel), 120 = manual

    active_choices = choices_pruned if COVERAGE_MODE == "pruned" else choices
    all_fields = list(active_choices.keys())
    sizes = [len(active_choices[f]) for f in all_fields]
    total_combo = int(np.prod(sizes))

    # Fingerprint of the option space (excluding volatile per-run paths). When the
    # grid changes, the fingerprint changes, so cached index->combo entries from an
    # incompatible space are simply treated as not-done instead of mis-attributed.
    _volatile = ("input_folder", "masks_folder", "output_folder")
    schema_fp = hashlib.sha1(
        json.dumps(
            [(f, [str(v) for v in active_choices[f]]) for f in all_fields if f not in _volatile],
            sort_keys=False,
        ).encode()
    ).hexdigest()[:12]

    # Initialize the db
    if RESTART:
        utils_test.initialize_db()
        if START_AT:
            utils_test.mark_hash_range_done(start=0, end=START_AT, namespace=COVERAGE_MODE)

    def _index_to_combo(flat_idx: int) -> tuple:
        """Decode a flat combo index into per-field indices (row-major)."""
        indices = []
        for s in reversed(sizes):
            indices.append(flat_idx % s)
            flat_idx //= s
        return tuple(reversed(indices))

    # Generate indices to visit.
    # Sampled: draw random indices directly (O(n_to_draw)), avoiding O(total_combo) traversal.
    # Exhaustive/pruned: stride so each shard processes exactly 1/SHARDS of all combos.
    if COVERAGE_MODE == "sampled":
        # Set PRNG seed per shard for reproducible, independent samples
        rng = random.Random(int(f'98234987234{SHARD_INDEX}'))
        n_to_draw = max(1, int(total_combo * PERCENT_SAMPLED / max(SHARDS, 1)))
        if n_to_draw < total_combo // 2:
            indices_to_process: Any = sorted(rng.sample(range(total_combo), n_to_draw))
        else:
            indices_to_process = sorted(set(rng.randint(0, total_combo - 1) for _ in range(n_to_draw)))
        n_indices = len(indices_to_process)
    else:
        indices_to_process = range(SHARD_INDEX, total_combo, SHARDS)
        n_indices = len(indices_to_process)

    pbar = None
    if SHOW_PROGRESS and tqdm is not None:
        pbar = tqdm(total=n_indices, initial=0, desc=f"[{COVERAGE_MODE}] Shard {SHARD_INDEX+1}/{SHARDS}", ncols=0)

    failures: list[str] = []      # hard failures: unexpected exceptions → test fails
    soft_failures: list[str] = []  # soft failures: reported but test passes
    # Coverage accounting (#7): every combo past START_AT lands in exactly one bucket.
    executed = 0        # ran the pipeline successfully
    skipped_cached = 0  # skipped because already terminal-success in the DB
    invalid = 0         # intentionally invalid option combination
    hard_failed = 0     # exception / dumped hard failure
    regressed = 0       # executed but produced soft anomalies
    cnt = -1
    for i in indices_to_process:
        if pbar is not None:
            pbar.update(1)

        # if `i` >= START_AT, proceed else next `i` (exhaustive/pruned only)
        if COVERAGE_MODE != "sampled" and i < START_AT:
            continue

        # Namespaced by coverage-mode + option-space fingerprint so neither the
        # different modes nor an evolving grid collide in the shared DB.
        combo_hash = f"{COVERAGE_MODE}:{schema_fp}:{i}"
        combo = tuple(active_choices[f][idx] for f, idx in zip(all_fields, _index_to_combo(i)))
        kwargs = dict(zip(all_fields, combo))
        cnt += 1

        # if combo never tested, proceed else next `i`
        if utils_test.is_already_done(combo_hash):
            skipped_cached += 1
            continue

        # Create target hist or spectrum if requested
        if (kwargs['mode'] == 9 and kwargs['dithering'] == 0) or (kwargs['mode'] == 9 and kwargs['legacy_mode']):
            # Register that combo as invalid
            invalid += 1
            utils_test.mark_hash_status(combo_hash, status='invalid_option_combination')
            continue

        try:
            opts = Options(**kwargs)
        except EXPECTED_INVALID as e:
            # Expected validation rejection → legitimately invalid combo
            invalid += 1
            utils_test.mark_hash_status(combo_hash, status='invalid', error=get_error_msg(e))
            continue
        except Exception as e:
            # Unexpected exception type while building Options → real bug, not "invalid"
            hard_failed += 1
            failures.append(
                f"\n💥 Options() raised unexpected {e.__class__.__name__} (combo {i})\n"
                f"→ Combo: {kwargs}\n"
                f"→ {e}\n"
            )
            utils_test.mark_hash_status(combo_hash, status='error', error=get_error_msg(e))
            continue

        ag = True if kwargs['legacy_mode'] else kwargs['as_gray']
        opts.target_hist = targets["hist"][int(ag)][kwargs['linear_luminance']][kwargs['rec_standard']] if kwargs['target_hist'] == "unit_test" else ("equal" if kwargs['target_hist'] == "equal" else None)
        # Precomputed spectra are at original image size; fft_padding_mode changes the expected
        # size, so let the pipeline compute its own target when padding is active.
        if kwargs['target_spectrum'] == 'unit_test' and kwargs.get('fft_padding_mode', 0) != 0:
            opts.target_spectrum = None
        else:
            opts.target_spectrum = targets["spec"][int(ag)][kwargs['linear_luminance']][kwargs['rec_standard']] if kwargs['target_spectrum'] == "unit_test" else None

        # Set seed. Derive it from a combo with volatile per-run path fields stripped
        # (e.g. the tmp masks_folder), otherwise the seed — and thus the pipeline output —
        # would change every run and reintroduce flakiness.
        if opts.seed is not None:
            seed_iter = opts.seed
        else:
            stable_combo = tuple(v for f, v in zip(all_fields, combo) if f not in _volatile)
            seed_iter = utils_test.deterministic_seed_from_combo(combo=stable_combo)

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
        try:
            opts.output_folder = out_dir
        except Exception as e:
            invalid += 1
            if out_dir and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            # Register that combo as invalid and set error as message from exception
            utils_test.mark_hash_status(combo_hash, status='invalid', error=get_error_msg(e))
            continue

        rand_selected_paths = None
        combo_warnings: list[str] = []
        try:

            # Options / pipeline
            rand_selected_images = utils_test.select_n_imgs(images_buffers['images'], n=2, seed=seed_iter)  # sRGB
            rand_selected_paths = utils_test.select_n_imgs(src_images_path, n=2, seed=seed_iter)  # sRGB

            ds = ImageDataset.model_construct(images=rand_selected_images, options=opts)
            # from_unit_test=True makes post_init skip the auto-run so the seed can be set
            # deterministically before processing; from_validation_test=True keeps the SSIM
            # validation recording enabled. This makes every combo reproducible and ensures
            # the dumped seed is the one actually used.
            proc = ImageProcessor.model_construct(
                dataset=ds, options=opts, verbose=-1,
                from_unit_test=True, from_validation_test=True,
            )
            proc.seed = seed_iter
            proc.process()

            # Prepare targets for validation
            th = opts.target_hist if proc._target_hist is None else proc._target_hist
            ts = opts.target_spectrum if proc._target_spectrum is None else proc._target_spectrum

            # Use the processor's own buffers so the before/after comparison shares the
            # exact color domain the pipeline produced (forward color treatment included).
            initial_buffers = proc._initial_buffer
            final_buffers = proc._final_buffer

            # internal validations — hard fail (crash = real bug)
            for rec in getattr(proc, "validation", []):
                if _is_fail(rec.get("valid_result", True)):
                    _dump_and_fail(rec, kwargs, seed_iter, rand_selected_paths, test_tmpdir)

            # --------- RMSE improvement checks for modes 5..8 ----------
            # Soft fail (strict):  RMSE must decrease (improve); any regression is noted.
            # Hard fail (lenient): RMSE more than doubled or increased > 0.1 — something badly wrong.
            _rmse_hard_tol = lambda b: np.maximum(0.1, b * 2) + 1e-9  # noqa: E731
            if opts.mode in (5, 6, 7, 8):
                # Histogram (always for 5..8)
                _, rmse_hist_before = utils.hist_match_validation(images=initial_buffers, binary_masks=proc.bool_masks, target_hist=th, normalize_rmse=True)
                _, rmse_hist_after = utils.hist_match_validation(images=final_buffers, binary_masks=proc.bool_masks, target_hist=th, normalize_rmse=True)
                if not np.all(rmse_hist_after <= rmse_hist_before + _rmse_hard_tol(rmse_hist_before)):
                    _dump_and_fail({"iter": -1, "step": -1, "processing_function": "hist_match", "valid_result": False,
                                    "log_result": f"Histogram RMSE more than doubled: {rmse_hist_before} -> {rmse_hist_after}"},
                                   kwargs, seed_iter, rand_selected_paths, test_tmpdir)
                elif not np.all(rmse_hist_after + 1e-9 <= rmse_hist_before):
                    combo_warnings.append(f"Histogram RMSE not improved: {rmse_hist_before} -> {rmse_hist_after}")

                # Spatial frequency (5,7)
                if opts.mode in (5, 7):
                    _, rmse_sf_before = utils.sf_match_validation(images=initial_buffers, target_spectrum=ts, normalize_rmse=True, fft_padding_mode=opts.fft_padding_mode, fft_padding_value=opts.fft_padding_value)
                    _, rmse_sf_after = utils.sf_match_validation(images=final_buffers, target_spectrum=ts, normalize_rmse=True, fft_padding_mode=opts.fft_padding_mode, fft_padding_value=opts.fft_padding_value)
                    if not np.all(rmse_sf_after <= rmse_sf_before + _rmse_hard_tol(rmse_sf_before)):
                        _dump_and_fail({"iter": -1, "step": -1, "processing_function": "sf_match", "valid_result": False,
                                        "log_result": f"SF RMSE more than doubled: {rmse_sf_before} -> {rmse_sf_after}"},
                                       kwargs, seed_iter, rand_selected_paths, test_tmpdir)
                    elif not np.all(rmse_sf_after + 1e-9 <= rmse_sf_before):
                        combo_warnings.append(f"SF RMSE not improved: {rmse_sf_before} -> {rmse_sf_after}")

                # Spectrum (6,8)
                if opts.mode in (6, 8):
                    _, rmse_spec_before = utils.spec_match_validation(images=initial_buffers, target_spectrum=ts, normalize_rmse=True, fft_padding_mode=opts.fft_padding_mode, fft_padding_value=opts.fft_padding_value)
                    _, rmse_spec_after = utils.spec_match_validation(images=final_buffers, target_spectrum=ts, normalize_rmse=True, fft_padding_mode=opts.fft_padding_mode, fft_padding_value=opts.fft_padding_value)
                    if not np.all(rmse_spec_after <= rmse_spec_before + _rmse_hard_tol(rmse_spec_before)):
                        _dump_and_fail({"iter": -1, "step": -1, "processing_function": "spec_match", "valid_result": False,
                                        "log_result": f"Spectrum RMSE more than doubled: {rmse_spec_before} -> {rmse_spec_after}"},
                                       kwargs, seed_iter, rand_selected_paths, test_tmpdir)
                    elif not np.all(rmse_spec_after + 1e-9 <= rmse_spec_before):
                        combo_warnings.append(f"Spectrum RMSE not improved: {rmse_spec_before} -> {rmse_spec_after}")

                # Recognizability safeguard on the FINAL composite output (not per-step
                # monotonicity). Two thresholds, mirroring the RMSE gate above:
                #   - hard (collapse): the image is structurally destroyed -> fail.
                #   - soft (degraded): still recognizable but quality is dropping -> warn.
                # Composite SSIM legitimately falls as iterations rise (the hist<->spectrum
                # constraints are incompatible and fight each other), so the hard floor stays
                # low to catch only a destroyed image, while the soft floor surfaces the
                # degraded regime where real problems begin. Both are env-tunable.
                ssim_collapse = float(os.getenv("COMPOSITE_SSIM_FLOOR", "0.3"))
                ssim_warn = float(os.getenv("COMPOSITE_SSIM_WARN", "0.6"))
                ssim_means = []
                for _idx in range(len(final_buffers)):
                    _, _ssim = utils.ssim_sens(initial_buffers[_idx], final_buffers[_idx],
                                               data_range=255, use_sample_covariance=False,
                                               binary_mask=proc.bool_masks[_idx])
                    ssim_means.append(float(np.mean(_ssim)))
                mean_ssim = float(np.mean(ssim_means))
                if mean_ssim < ssim_collapse:
                    _dump_and_fail({"iter": -1, "step": -1, "processing_function": "composite", "valid_result": False,
                                    "log_result": f"Final composite SSIM {mean_ssim:.4f} collapsed below {ssim_collapse} (per-image {np.round(ssim_means, 4)})"},
                                   kwargs, seed_iter, rand_selected_paths, test_tmpdir)
                elif mean_ssim < ssim_warn:
                    combo_warnings.append(f"Composite SSIM degraded: {mean_ssim:.4f} < {ssim_warn} (per-image {np.round(ssim_means, 4)})")
            # ---------------------------------------------------------------------

            # SSIM optimization checks — per-hist_match passes.
            # The 'final' pass checks SSIM(first proposal -> hist_match output). This is a
            # meaningful *terminal* check only for mode 2, where hist_match is the last op.
            # In composite modes (5-8) hist_match is intermediate — sf_match/spec_match
            # deliberately move the image afterwards — so per-step SSIM monotonicity is not
            # psychophysically meaningful; it is demoted to diagnostic and the composite
            # end-state SSIM floor below is the real recognizability gate.
            if opts.hist_optim and opts.mode in (2, 5, 6, 7, 8):
                for rec in getattr(proc, "ssim_results", []):
                    if _is_fail(rec.get("valid_result", True)):
                        tag = rec.get('tag', 'sub_iter')
                        msg = (
                            f"SSIM optimization {tag} regression: "
                            f"image={rec.get('image')}, channel={rec.get('channel')}, "
                            f"iter={rec.get('iter')}, step={rec.get('step')}, "
                            f"ssim_values={rec.get('ssim_values')}"
                        )
                        if tag == 'final' and opts.mode == 2:
                            _dump_and_fail({**rec, "log_result": msg}, kwargs, seed_iter, rand_selected_paths, test_tmpdir)
                        else:
                            combo_warnings.append(msg)

        except ControlledFailure as cf:
            hard_failed += 1
            failures.append(str(cf))
            utils_test.mark_hash_status(combo_hash, status='failed')
        except Exception as e:
            hard_failed += 1
            tb = format_exc()
            dump_path = utils_test.dump_failure_context(
                combo_dict=kwargs,
                rec={"iter": 0, "step": 0, "error": str(e), "traceback": tb},
                tmp_root=test_tmpdir,
                seed=seed_iter,
                selected_paths=rand_selected_paths,
                file_type=DUMP_FILE_FORMAT,
            )
            failures.append(
                f"\n💥 Unexpected error\n"
                f"→ Shard {SHARD_INDEX}, Combo global-index {i}, Per-shard #{cnt}\n"
                f"→ Combo: {kwargs}\n"
                f"→ Exception: {e.__class__.__name__}: {e}\n"
                f"→ Dumped context: {dump_path}\n"
            )
            utils_test.mark_hash_status(combo_hash, status='error')
        else:
            # Pipeline ran to completion for this combo.
            executed += 1
            # Cleanup safely
            if out_dir and out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            if combo_warnings:
                regressed += 1
                warn_str = "; ".join(combo_warnings)
                dump_path = utils_test.dump_failure_context(
                    combo_dict=kwargs,
                    rec={"iter": -1, "step": -1, "warnings": combo_warnings},
                    tmp_root=test_tmpdir,
                    seed=seed_iter,
                    selected_paths=rand_selected_paths or [],
                    file_type=DUMP_FILE_FORMAT,
                )
                entry = (
                    f"\nSoft failure — Combo {i} mode={opts.mode} hist_optim={opts.hist_optim}\n"
                    f"  {warn_str}\n"
                    f"  → Dumped: {dump_path}"
                )
                if STRICT_SOFT:
                    # Promote to a hard failure and keep it red on re-run until fixed.
                    failures.append(entry)
                    utils_test.mark_hash_status(combo_hash, status='failed', error=warn_str)
                else:
                    soft_failures.append(entry)
                    utils_test.mark_hash_status(combo_hash, status='done', error=warn_str)
            else:
                utils_test.mark_hash_status(combo_hash, status='done')

    if pbar is not None:
        pbar.close()
    if soft_failures:
        print(f"\n[soft failures] {len(soft_failures)} combo(s) reported optimization anomalies:")
        for sf in soft_failures:
            print(f"  {sf}")

    # ----- coverage accounting (#7) -----
    # Every combo that passed the START_AT filter (cnt + 1 of them) must land in
    # exactly one bucket; otherwise a combo was silently dropped.
    total_seen = cnt + 1
    bucketed = executed + skipped_cached + invalid + hard_failed
    print(
        f"\n[coverage] seen={total_seen} executed={executed} cached={skipped_cached} "
        f"invalid={invalid} hard_failed={hard_failed} regressed={regressed}"
    )
    assert bucketed == total_seen, (
        f"combo accounting mismatch: {bucketed} bucketed != {total_seen} seen "
        "(a combo was silently dropped)"
    )
    # Prove the shard did real work this run (ran/evaluated at least one combo) —
    # unless every combo was legitimately cached from a previous run.
    assert (executed + hard_failed + invalid) > 0 or skipped_cached == total_seen, (
        "no combos were evaluated and not all were cached — the run did no real work"
    )

    # ----- aggregate soft-failure gate (#5) -----
    soft_rate = regressed / max(executed, 1)
    assert soft_rate <= MAX_SOFT_FAIL_RATE, (
        f"soft-failure (regression) rate {soft_rate:.2%} exceeds "
        f"MAX_SOFT_FAIL_RATE={MAX_SOFT_FAIL_RATE:.2%}"
    )

    assert not failures, f"{len(failures)} combo(s) failed:\n" + "\n---\n".join(failures)


# -------------------- helpers --------------------
def _is_fail(valid_result: Any) -> bool:
    """Normalize the mixed ``valid_result`` field into a failure boolean.

    ``valid_result`` can be a bool / numpy bool (from ``_validate``) or one of the
    strings ``"PASS"`` / ``"WARN"`` / ``"FAIL"`` (from diagnostic records). Only
    a boolean-False or the explicit ``"FAIL"`` string counts as a failure;
    ``"WARN"`` is a diagnostic and must not fail the test.
    """
    if isinstance(valid_result, str):
        return valid_result == "FAIL"
    return not bool(valid_result)


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


def get_error_msg(e):
    exc_type = e.__class__.__name__
    msg = str(e)
    if msg:
        error_msg = f"[{exc_type}]: {msg}"
    else:
        error_msg = f"[{exc_type}]"

    return error_msg
