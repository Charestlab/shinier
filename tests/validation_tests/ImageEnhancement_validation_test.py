from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from shinier.color.Converter import rgb2gray
from shinier.utils import (
    MatlabOperators,
    betce_gray,
    compute_ambe,
    compute_bp2bpsim,
    compute_contrast_improvement,
    compute_image_entropy,
    compute_mssim,
    compute_psnr,
    nfldice_gray,
    rdfhe_gray,
    sfcef_gray,
    tidhe_gray,
)

pytestmark = pytest.mark.validation_tests

# MATLAB reference hashes were generated from the free imTIDHE/imRDFHE/imNFLDICE/imBETCE/imSFCEF
# implementations linked in their respective articles; see tests/README.md.
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
INPUT_DIR = ASSET_DIR / "SAMPLE_512X512"
SFCEF_MATLAB_DIR = ASSET_DIR / "sfcef_matlab_reference"
MATLAB_HASHES = json.loads((ASSET_DIR / "image_enhancement_matlab_sha256.json").read_text())
ALGORITHMS = {"tidhe": tidhe_gray, "rdfhe": rdfhe_gray, "nfldice": nfldice_gray, "betce": betce_gray, "sfcef": sfcef_gray}
# SFCEF excluded: MATLAB filter2 uses FMA; pixel-level parity is tested separately via max_diff + exact diff count.
STRICT_HASH_ALGORITHMS = {"tidhe", "rdfhe", "nfldice", "betce"}
IMAGE_INDEX = {name: idx for idx, name in enumerate(MATLAB_HASHES["images"])}


def _matlab_gray(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return MatlabOperators.uint8(rgb2gray(rgb, weighting_standard="rec601", matlab_601=True))


def _hash_output(algorithm: str, input_path: Path) -> dict[str, object]:
    actual = ALGORITHMS[algorithm](_matlab_gray(input_path))
    expected_hash = MATLAB_HASHES["sha256"][algorithm][IMAGE_INDEX[input_path.name]]
    return {
        "algorithm": algorithm,
        "image": input_path.name,
        "shape": list(actual.shape),
        "expected_shape": MATLAB_HASHES["shape"],
        "dtype": str(actual.dtype),
        "expected_dtype": MATLAB_HASHES["dtype"],
        "sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
        "expected_sha256": expected_hash,
    }


@pytest.mark.parametrize("algorithm", STRICT_HASH_ALGORITHMS)
@pytest.mark.parametrize("input_path", sorted(INPUT_DIR.glob("*.png")), ids=lambda p: p.stem)
def test_image_enhancement_matches_matlab_reference_hash(algorithm: str, input_path: Path) -> None:
    row = _hash_output(algorithm, input_path)
    assert row["shape"] == row["expected_shape"], row
    assert row["dtype"] == row["expected_dtype"], row
    assert row["sha256"] == row["expected_sha256"], row


_SFCEF_SYNTHETIC_PATTERNS: dict[str, "np.ndarray"] = {}


def _sfcef_synthetic_inputs() -> dict[str, np.ndarray]:
    """Three 1000×1000 synthetic patterns, all in [0, 32].

    All pixels ≤ 32 guarantees 0% clipping in sfcef_gray (max output = 7.8×32 = 249.6 < 255),
    exercising the full output range — unlike natural images where 80%+ of pixels saturate to 255.
    Three gradient directions stress the filter in orthogonal orientations.
    """
    if _SFCEF_SYNTHETIC_PATTERNS:
        return _SFCEF_SYNTHETIC_PATTERNS
    n = 1000
    ii, jj = np.mgrid[0:n, 0:n]
    _SFCEF_SYNTHETIC_PATTERNS["sfcef_synth_diag"] = np.round(32.0 * (ii + jj) / (999 + 999)).astype(np.uint8)
    _SFCEF_SYNTHETIC_PATTERNS["sfcef_synth_horiz"] = np.round(32.0 * jj / 999).astype(np.uint8)
    _SFCEF_SYNTHETIC_PATTERNS["sfcef_synth_vert"] = np.round(32.0 * ii / 999).astype(np.uint8)
    return _SFCEF_SYNTHETIC_PATTERNS


@pytest.mark.parametrize("pattern_name", ["sfcef_synth_diag", "sfcef_synth_horiz", "sfcef_synth_vert"])
def test_sfcef_pixel_diff_from_matlab(pattern_name: str) -> None:
    """SFCEF output on a synthetic low-value image must match MATLAB imSFCEF within FMA tolerance.

    Inputs: three 1000×1000 gradients in [0, 32] (zero clipping, full output range exercised).
    Two assertions per pattern:
    - max_diff <= 1: no pixel deviates by more than 1 gray level (FMA theoretical bound).
    - n_diffs == expected: exact count of differing pixels, catching systematic shifts.
    """
    gray = _sfcef_synthetic_inputs()[pattern_name]
    matlab_ref = np.asarray(Image.open(SFCEF_MATLAB_DIR / f"{pattern_name}_matlab.png").convert("L"))
    python_out = sfcef_gray(gray, legacy_mode=True)
    diff = np.abs(python_out.astype(np.int16) - matlab_ref.astype(np.int16))
    n_diffs = int(np.sum(diff > 0))
    expected_n_diffs = MATLAB_HASHES["sfcef_synthetic_n_diffs"][pattern_name]
    total_pixels = gray.size
    print(f"\n  {pattern_name}: {n_diffs}/{total_pixels} pixels off ({100 * n_diffs / total_pixels:.6f}%), max_diff={int(diff.max())}")
    assert diff.max() <= 1, f"max pixel diff = {diff.max()} (expected ≤ 1)"
    assert n_diffs == expected_n_diffs, (
        f"{n_diffs} differing pixels (expected {expected_n_diffs}). "
        "A systematic shift would produce a much larger count."
    )


@pytest.mark.parametrize("pattern_name", ["sfcef_synth_diag", "sfcef_synth_horiz", "sfcef_synth_vert"])
def test_sfcef_metric_diff_from_matlab_under_point_one_percent(pattern_name: str) -> None:
    """SFCEF validation metrics must stay within 0.1% of MATLAB values.

    Pairwise metrics use the synthetic input as the reference image. CI and
    entropy are computed directly on each enhanced output.
    """
    gray = _sfcef_synthetic_inputs()[pattern_name]
    matlab_ref = np.asarray(Image.open(SFCEF_MATLAB_DIR / f"{pattern_name}_matlab.png").convert("L"))
    python_out = sfcef_gray(gray, legacy_mode=True)
    metrics = {
        "ambe_vs_input": (compute_ambe(gray, python_out), compute_ambe(gray, matlab_ref)),
        "mssim_vs_input": (compute_mssim(gray, python_out, data_range=255), compute_mssim(gray, matlab_ref, data_range=255)),
        "psnr_vs_input": (compute_psnr(gray, python_out, data_range=255), compute_psnr(gray, matlab_ref, data_range=255)),
        "bp2bpsim_vs_input": (compute_bp2bpsim(gray, python_out), compute_bp2bpsim(gray, matlab_ref)),
        "ci": (compute_contrast_improvement(python_out), compute_contrast_improvement(matlab_ref)),
        "entropy": (compute_image_entropy(python_out), compute_image_entropy(matlab_ref)),
    }
    rows = {}
    for name, (python_value, matlab_value) in metrics.items():
        abs_diff = float(np.abs(python_value - matlab_value))
        percent_of_matlab = 0.0 if matlab_value == 0 and abs_diff == 0 else abs_diff / float(np.abs(matlab_value)) * 100.0
        rows[name] = {
            "python": float(python_value),
            "matlab": float(matlab_value),
            "abs_diff": abs_diff,
            "percent_of_matlab": percent_of_matlab,
        }
    print(f"\n  {pattern_name} metric diff vs MATLAB: {rows}")
    assert all(row["percent_of_matlab"] <= 0.1 for row in rows.values()), rows


def test_image_enhancement_matlab_comparison_report(tmp_path: Path) -> None:
    rows = [
        _hash_output(algorithm, path)
        for algorithm in ALGORITHMS
        for path in sorted(INPUT_DIR.glob("*.png"))
    ]
    report = tmp_path / "image_enhancement_matlab_python_hash_comparison.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert all(
        row["sha256"] == row["expected_sha256"]
        for row in rows
        if row["algorithm"] in STRICT_HASH_ALGORITHMS
    ), rows
