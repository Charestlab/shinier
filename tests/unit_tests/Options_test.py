"""
Comprehensive validation tests for shinier.Options.

Covers:
    1. Generic schema-driven validation for all fields (valid/invalid)
    2. Specific cross-field consistency and behavioral logic
    3. Edge cases and boundary conditions
"""
from tqdm.auto import tqdm
import itertools
import pytest
import os
import random
import numpy as np
from pathlib import Path
from pydantic import ValidationError
from shinier.Options import Options
from tests.validation_tests.ImageProcessor_validation_test import get_possible_values
from tests.tools import utils as utils_test
# pytestmark = pytest.mark.unit_tests

# -------------------- env --------------------
START_AT = int(os.getenv("START_AT", "0"))
SHARDS = int(os.getenv("SHARDS", "1"))
SHARD_INDEX = int(os.getenv("SHARD_INDEX", "0"))
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "1") == "1"
PERCENT_SAMPLED = float(os.getenv("PERCENT_SAMPLED", "1"))

# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary folders that always exist for valid path tests."""
    in_dir = tmp_path / "INPUT"
    out_dir = tmp_path / "OUTPUT"
    mask_dir = tmp_path / "MASKS"
    for d in (in_dir, out_dir, mask_dir):
        d.mkdir()
    return in_dir, out_dir, mask_dir


# =============================================================================
# GENERIC FIELD VALIDATION (Schema-Driven)
# =============================================================================
def generate_valid_kwargs(tmp_dirs):
    """Return valid defaults merged with temporary directories."""
    in_dir, out_dir, mask_dir = tmp_dirs
    valid = {
        name: field.default
        for name, field in Options.model_fields.items()
        if field.default is not None
    }
    valid.update(dict(input_folder=in_dir, output_folder=out_dir, masks_folder=mask_dir))
    return valid


def generate_invalid_value(field):
    """Heuristically produce an invalid value for a given field."""
    t = field.annotation

    # Literal types (e.g. Literal[0, 1, 2])
    if getattr(t, "__origin__", None) is not None and "Literal" in str(t.__origin__):
        valid = t.__args__
        if isinstance(valid[0], int):
            return max(valid) + 999
        if isinstance(valid[0], str):
            return "INVALID_LITERAL"
        return object()

    # Constrained numeric types
    if "conint" in str(t):
        return -999
    if "confloat" in str(t):
        return -9.99

    # Simple primitives
    if t in (int, float):
        return "not_a_number"
    if t is bool:
        return "maybe"

    # Paths
    if "Path" in str(t):
        return Path("/nonexistent/folder")

    # Arrays
    if "ndarray" in str(t):
        return np.ones((8, 8), dtype=int)  # invalid dtype (should be float)

    # Fallback
    return object()


def generate_invalid_kwargs(field_name):
    """Generate dict containing an invalid entry for a given field."""
    f = Options.model_fields[field_name]
    return {field_name: generate_invalid_value(f)}


@pytest.mark.unit_tests
@pytest.mark.parametrize("field_name", list(Options.model_fields))
def test_field_validation(field_name, tmp_dirs):
    """Schema-based validation for every field in the model."""
    valid = generate_valid_kwargs(tmp_dirs)
    # -- should pass with all defaults --
    _ = Options(**valid)

    # -- mutate with invalid value and expect error --
    invalid = generate_invalid_kwargs(field_name)
    valid.update(invalid)
    with pytest.raises((ValueError, TypeError, ValidationError)):
        Options(**valid)


# =============================================================================
# SPECIFIC BEHAVIORAL & CROSS-FIELD TESTS
# =============================================================================
@pytest.mark.unit_tests
def test_default_initialization(tmp_dirs):
    """Ensure defaults instantiate correctly."""
    in_dir, out_dir, _ = tmp_dirs
    opt = Options(input_folder=in_dir, output_folder=out_dir)
    assert opt.mode == 2
    assert opt.rescaling == 0
    assert opt.background == 300
    assert opt.hist_specification == 4
    assert isinstance(opt.as_gray, bool)


@pytest.mark.unit_tests
def test_invalid_paths_raise(tmp_path):
    """Non-existent folders must raise ValueError."""
    bogus = tmp_path / "nonexistent"
    with pytest.raises(ValueError):
        Options(input_folder=bogus, output_folder=bogus)


@pytest.mark.unit_tests
@pytest.mark.parametrize("val", [-1, 400, 999])
def test_background_out_of_range(val, tmp_dirs):
    """Background intensity must be [0–255] or 300."""
    in_dir, out_dir, _ = tmp_dirs
    with pytest.raises((ValidationError, ValueError)):
        Options(input_folder=in_dir, output_folder=out_dir, background=val)

@pytest.mark.unit_tests
def test_background_valid_values(tmp_dirs):
    """0, 255, and 300 are valid backgrounds."""
    in_dir, out_dir, _ = tmp_dirs
    for val in (0, 255, 300):
        opt = Options(input_folder=in_dir, output_folder=out_dir, background=val)
        assert opt.background == val


@pytest.mark.unit_tests
def test_target_hist_valid_and_invalid_shapes(tmp_dirs):
    """Validate histogram array shape and dtype constraints."""
    in_dir, out_dir, _ = tmp_dirs
    valid_1d = np.zeros((256,))
    valid_3ch = np.zeros((256, 3))
    _ = Options(input_folder=in_dir, output_folder=out_dir, target_hist=valid_1d)
    _ = Options(input_folder=in_dir, output_folder=out_dir, target_hist=valid_3ch, linear_luminance=True)

    bad_shape = np.zeros((128,))
    with pytest.raises(ValueError):
        Options(input_folder=in_dir, output_folder=out_dir, target_hist=bad_shape)

    bad_type = "notarray"
    with pytest.raises((TypeError, ValidationError)):
        Options(input_folder=in_dir, output_folder=out_dir, target_hist=bad_type)


@pytest.mark.unit_tests
def test_target_spectrum_validation(tmp_dirs):
    """Target spectrum must be float ndarray."""
    in_dir, out_dir, _ = tmp_dirs
    good = np.ones((16, 16), dtype=float)
    bad_type = np.ones((16, 16), dtype=int)
    _ = Options(input_folder=in_dir, output_folder=out_dir, target_spectrum=good)
    with pytest.raises(TypeError):
        Options(input_folder=in_dir, output_folder=out_dir, target_spectrum=bad_type)


@pytest.mark.unit_tests
def test_hist_optim_overwrites_hist_spec(tmp_dirs):
    """hist_optim=True should nullify hist_specification."""
    in_dir, out_dir, _ = tmp_dirs
    opt = Options(input_folder=in_dir, output_folder=out_dir, hist_optim=True)
    assert opt.hist_specification is None


@pytest.mark.unit_tests
def test_rescaling_forbidden_modes_overwrite(tmp_dirs):
    """Rescaling forced to 0 for luminance/histogram modes."""
    in_dir, out_dir, _ = tmp_dirs
    for mode in (1, 2):
        opt = Options(input_folder=in_dir, output_folder=out_dir, mode=mode, rescaling=2)
        assert opt.rescaling == 0


@pytest.mark.unit_tests
def test_mode9_dithering_zero_raises(tmp_dirs):
    """Mode 9 cannot have dithering=0."""
    in_dir, out_dir, _ = tmp_dirs
    with pytest.raises(ValueError):
        Options(input_folder=in_dir, output_folder=out_dir, mode=9, dithering=0)


@pytest.mark.unit_tests
def test_iterations_clamped_for_noncomposite_modes(tmp_dirs):
    """iterations >1 only valid for composite modes (5–8)."""
    in_dir, out_dir, _ = tmp_dirs
    opt = Options(input_folder=in_dir, output_folder=out_dir, mode=3, iterations=5)
    assert opt.iterations == 1


@pytest.mark.unit_tests
def test_whole_image_requires_mask_folder(tmp_dirs):
    """whole_image=3 requires masks_folder."""
    in_dir, out_dir, mask_dir = tmp_dirs
    # valid
    _ = Options(input_folder=in_dir, output_folder=out_dir, whole_image=3, masks_folder=mask_dir)
    # missing folder → fail
    with pytest.raises(ValueError):
        Options(input_folder=in_dir, output_folder=out_dir, whole_image=3, masks_folder=None)


@pytest.mark.unit_tests
def test_legacy_mode_overrides(tmp_dirs):
    """legacy_mode should force multiple field overwrites."""
    in_dir, out_dir, _ = tmp_dirs
    opt = Options(input_folder=in_dir, output_folder=out_dir, legacy_mode=True)
    assert not opt.conserve_memory
    assert opt.as_gray
    assert opt.dithering == 0
    assert opt.hist_specification == 1
    assert not opt.safe_lum_match


@pytest.mark.unit_tests
def test_export_schema(tmp_dirs, tmp_path):
    """Ensure schema export works."""
    in_dir, out_dir, _ = tmp_dirs
    opt = Options(input_folder=in_dir, output_folder=out_dir)
    schema_path = tmp_path / "schema.json"
    opt.export_schema(schema_path)
    assert schema_path.exists()
    txt = schema_path.read_text()
    assert "title" in txt and "properties" in txt


@pytest.mark.unit_tests
def test_repr_output(tmp_dirs):
    """__repr__ should contain all key-value pairs."""
    in_dir, out_dir, _ = tmp_dirs
    opt = Options(input_folder=in_dir, output_folder=out_dir)
    r = repr(opt)
    assert "mode:" in r and "input_folder" in r


@pytest.mark.test_all_options
def test_all_combo(tmp_dirs):
    mask_dir = tmp_dirs[2]
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
    total_combo = np.prod([len(v) for v in choices.values() if hasattr(v, '__len__') and not isinstance(v, str)])

    pbar = None
    if SHOW_PROGRESS and tqdm is not None:
        per_shard = total_combo // SHARDS + (1 if SHARD_INDEX < (total_combo % SHARDS) else 0)
        pbar = tqdm(total=total_combo, initial=START_AT, desc=f"Shard {SHARD_INDEX+1}/{SHARDS}", ncols=0)

    # Set PNRG seed for test sampling
    rng = random.Random()  # independent state
    rng.seed(int(f'98234987234{SHARD_INDEX}'))

    keys = list(choices)
    for i, combo in enumerate(itertools.product(*(choices[k] for k in keys))):
        if pbar is not None:
            pbar.update(1)

        # if `i` within this SHARD_INDEX, proceed else next `i`
        if i % SHARDS != SHARD_INDEX:
            continue

        # if `i` >= START_ITER, proceed else next `i`
        if i < START_AT:
            continue

        # if not randomly selected, proceed else next `i`
        is_sampled = rng.random() >= (1 - PERCENT_SAMPLED)
        if not is_sampled:
            continue

        params = dict(zip(keys, combo))
        if params['mode'] == 9 and params['dithering'] == 0:
            params['dithering'] = 1
        Options(**params)
        pbar.update(1)
