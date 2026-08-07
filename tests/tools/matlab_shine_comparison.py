"""Compare SHINIER (Python) against the original MATLAB SHINE toolbox.

Run the complete comparison with:

    bash tests/tools/run_matlab_shine_comparison.sh

Requirements: MATLAB and the SHINE toolbox
(http://www.mapageweb.umontreal.ca/gosselif/SHINE/).

Three implementations are compared across processing modes 1-8:

- ``matlab_shine``: the original MATLAB SHINE toolbox (lumMatch, histMatch,
  sfMatch, specMatch), driven by a generated MATLAB script that replicates
  the ``SHINE.m`` composite-mode loop;
- ``shinier_legacy``: SHINIER with ``legacy_mode=True`` (MATLAB-compatible
  grayscale conversion, rounding, and noise tie-breaking);
- ``shinier_modern_gray``: SHINIER defaults on grayscale (xyY luminance
  processing, hybrid tie-breaking, sRGB-encoded export).

Two comparison stages are produced:

1. Output comparison: pixel differences between saved MATLAB and Python
   images (RMSE, MAE, max abs, equal fraction, histogram L1).
2. Fixed-target comparison: every implementation receives the same fixed
   initial Python targets (histogram and spectrum), and each output is
   measured against the target in its own processing domain. MATLAB and
   ``shinier_legacy`` share the legacy target; ``shinier_modern_gray`` uses
   its own target.

Default output is CSV-only; use ``FULL_TRACKING=1`` or ``--full-tracking`` to
keep generated inputs, PNGs, MATLAB scripts, and ``.mat`` files. The grayscale
CSV compares MATLAB's input gray image to SHINIER's internal
``ImageProcessor._initial_buffer``. PNG target metrics map saved images back
through each implementation's grayscale domain; ``png_hist_l1`` also uses
``rounded_target_hist`` and the export/import bin transform. SHINIER rows
include ``soft_clip``; MATLAB rows use SHINE's rescale/uint8 behavior.
Terminal tables use scientific notation with 3 decimals; CSVs keep full
precision.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHINE_DIR = Path(os.environ["SHINE_DIR"]) if os.environ.get("SHINE_DIR") else None
DEFAULT_MATLAB = Path(os.environ.get("MATLAB_BIN", "matlab"))
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp/matlab_shine_comparison"

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "shinier_matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

sys.path.insert(0, str(REPO_ROOT / "src"))

from shinier import ImageDataset, ImageProcessor, Options  # noqa: E402
from shinier.color.Converter import ColorConverter, rgb2gray  # noqa: E402
from shinier.utils import (  # noqa: E402
    _crop_after_fft,
    compute_rmse,
    compute_tvd_hist,
    get_radius_grid,
    image_spectrum,
    MatlabOperators,
    pol2cart,
    rounded_target_hist,
    rotational_avg,
    soft_clip,
    uint8_plus,
)

TARGET_REFERENCE = (
    "Initial Python target. MATLAB and SHINIER legacy use the legacy target; "
    "SHINIER modern-gray uses its own modern-gray target. Composite modes are "
    "evaluated against this fixed initial target. In this target-injection run, "
    "MATLAB and Python are given the fixed target; normal composite processing "
    "without injected targets may use moving targets internally."
)
ANSI_BOLD_RED = "\033[1;31m"
ANSI_RESET = "\033[0m"
MATLAB_IMPLEMENTATION = "matlab_shine"
LEGACY_IMPLEMENTATION = "shinier_legacy"
MODERN_GRAY_IMPLEMENTATION = "shinier_modern_gray"
PYTHON_IMPLEMENTATIONS = {
    LEGACY_IMPLEMENTATION: ("legacy", True),
    MODERN_GRAY_IMPLEMENTATION: ("modern_gray", False),
}

HIST_MODES = {2, 5, 6, 7, 8}
SF_MODES = {3, 5, 7}
SPECTRUM_MODES = {4, 6, 8}
TARGET_MODES = HIST_MODES | SF_MODES | SPECTRUM_MODES

TARGET_METRIC_LABELS = {
    "internal_buffer_hist_l1_to_python_target": "int_hist_l1",
    "exported_png_hist_l1_to_python_target": "png_hist_l1",
    "internal_buffer_sf_rmse_to_python_target": "int_sf_rmse",
    "exported_png_sf_rmse_to_python_target": "png_sf_rmse",
    "pre_range_fourier_spectrum_rmse_to_python_target": "fft_spec",
    "pre_range_image_spectrum_rmse_to_python_target": "pre_spec",
    "pre_range_out_of_range_fraction": "pre_oor",
    "post_soft_clip_spectrum_rmse_to_python_target": "clip_spec",
    "internal_buffer_spectrum_rmse_to_python_target": "int_spec_rmse",
    "exported_png_spectrum_rmse_to_python_target": "png_spec_rmse",
}
PIXEL_COLUMNS = "images mean_rmse mean_mae max_abs mean_equal_fraction mean_hist_l1".split()
OUTPUT_COMPARISON_COLUMNS = "mode reference python_output".split() + PIXEL_COLUMNS[1:]
TARGET_TABLE_COLUMNS = ["mode", "implementation", "target"] + [
    f"mean_{key}" for key in TARGET_METRIC_LABELS
]
TARGET_TABLE_LABELS = {f"mean_{key}": label for key, label in TARGET_METRIC_LABELS.items()}
# CSV-only keys extend the displayed metrics with the signed out-of-range fractions.
TARGET_METRIC_KEYS = (
    *TARGET_METRIC_LABELS,
    "pre_range_below_zero_fraction",
    "pre_range_above_one_fraction",
)
TARGET_TABLE_NOTES = [
    "Legend:",
    "  int_* = metric on SHINIER ImageProcessor._final_buffer; blank for MATLAB.",
    "  png_* = exported PNG mapped back to the implementation target domain.",
    "  fft/pre/clip_spec = SHINIER-only spectrum checks before/after soft_clip; pre_oor = pre-correction out-of-range fraction.",
    "  clip_spec vs int_spec_rmse gap = final rescaling for modes 4/6, final hist_match for mode 8.",
    "  SHINIER uses soft_clip, which lowers hard clipping but can worsen fixed-target RMSE; MATLAB uses SHINE rescale/uint8.",
    "  png_hist_l1 uses rounded_target_hist plus the same export/import bin transform.",
]
TARGET_WARNING_LINES = [
    "target metrics use fixed initial Python targets; normal composite runs may use moving targets internally.",
    "MATLAB/legacy share the legacy target; modern-gray uses its own target.",
    "SHINIER rows include soft_clip; MATLAB rows use SHINE rescale/uint8.",
    "png_hist_l1 uses rounded_target_hist plus the export/import bin mapping.",
]
RUN_INFO_KEYS = [
    "run_root",
    "matlab_runner",
    "modes",
    "images",
    "use_python_targets",
    "skip_matlab",
    "prepare_only",
]
INPUT_GRAY_TITLE = "MATLAB input grayscale vs SHINIER internal initial buffer"
TargetSet = dict[str, np.ndarray | int | str | bool]
TargetSets = dict[str, TargetSet]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add = parser.add_argument
    add("--input-dir", type=Path, default=REPO_ROOT / "tests/assets/SAMPLE_64X64")
    add("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add("--run-root", type=Path)
    add("--shine-dir", type=Path, default=DEFAULT_SHINE_DIR)
    add("--matlab-bin", type=Path, default=DEFAULT_MATLAB)
    add("--modes", type=int, nargs="+", default=list(range(1, 9)))
    add("--iterations", type=int, default=5)
    add("--seed", type=int, default=42)
    add("--limit", type=int, default=8)
    for flag in (
        "--skip-matlab",
        "--prepare-only",
        "--use-python-targets",
        "--quiet-run-info",
        "--full-tracking",
    ):
        add(flag, action="store_true")
    return parser.parse_args()


def quote_matlab_path(path: Path) -> str:
    return str(path).replace("'", "''")


def prepare_inputs(source_dir: Path, output_root: Path, limit: int) -> list[Path]:
    input_dir = output_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(source_dir.glob("*.png"))[:limit]
    if not paths:
        raise FileNotFoundError(f"No PNG inputs found in {source_dir}")
    for path in paths:
        shutil.copy2(path, input_dir / path.name)
    return [input_dir / path.name for path in paths]


def write_matlab_runner(
    *,
    output_root: Path,
    input_dir: Path,
    shine_dir: Path | None,
    modes: list[int],
    iterations: int,
    seed: int,
    use_python_targets: bool = False,
) -> Path:
    runner = output_root / "run_matlab_shine.m"
    modes_text = " ".join(str(mode) for mode in modes)
    shine_addpath = f"addpath('{quote_matlab_path(shine_dir)}');" if shine_dir else ""
    shine_check = (
        "required_shine = {'lumMatch', 'histMatch', 'sfMatch', 'specMatch'};\n"
        "missing_shine = {};\n"
        "for si = 1:numel(required_shine)\n"
        "    if exist(required_shine{si}, 'file') ~= 2\n"
        "        missing_shine{end + 1} = required_shine{si}; %#ok<AGROW>\n"
        "    end\n"
        "end\n"
        "if ~isempty(missing_shine)\n"
        "    error(['Missing SHINE functions: ', strjoin(missing_shine, ', '), ...\n"
        "        '. Set SHINE_DIR to the SHINE toolbox folder or download SHINE from ', ...\n"
        "        'http://www.mapageweb.umontreal.ca/gosselif/SHINE/']);\n"
        "end"
    )
    target_load = ""
    hist_arg = ""
    spectrum_arg = ""
    if use_python_targets:
        target_load = (
            f"targets = load('{quote_matlab_path(output_root / 'python_targets.mat')}');"
        )
        hist_arg = ", targets.target_hist_counts"
        spectrum_arg = ", targets.target_spectrum"
    runner.write_text(
        textwrap.dedent(
            f"""
            {shine_addpath}
            {shine_check}
            input_dir = '{quote_matlab_path(input_dir)}';
            output_root = '{quote_matlab_path(output_root / "matlab")}';
            modes = [{modes_text}];
            {target_load}
            files = dir(fullfile(input_dir, '*.png'));
            [~, order] = sort({{files.name}});
            files = files(order);

            original = cell(1, numel(files));
            input_gray_dir = fullfile(output_root, 'input_gray');
            if ~exist(input_gray_dir, 'dir')
                mkdir(input_gray_dir);
            end
            for k = 1:numel(files)
                im = imread(fullfile(input_dir, files(k).name));
                if ndims(im) == 3
                    im = rgb2gray(im);
                end
                original{{k}} = im;
                imwrite(uint8(im), fullfile(input_gray_dir, files(k).name));
            end

            hist_corr_summary = {{}};
            for mi = 1:numel(modes)
                mode = modes(mi);
                rand('seed', {seed});
                images = original;
                if mode >= 5
                    n_iter = {iterations};
                else
                    n_iter = 1;
                end
                for iter = 1:n_iter
                    switch mode
                        case 1
                            images = lumMatch(images);
                        case {{2, 5, 6}}
                            [shine_log, images] = evalc('histMatch(images, 0{hist_arg})');
                            hist_corr_summary = append_hist_corr( ...
                                hist_corr_summary, mode, iter, 'hist_before_spectrum', shine_log);
                    end
                    switch mode
                        case {{3, 5, 7}}
                            images = sfMatch(images, 1{spectrum_arg});
                        case {{4, 6, 8}}
                            images = specMatch(images, 1{spectrum_arg});
                    end
                    switch mode
                        case {{7, 8}}
                            [shine_log, images] = evalc('histMatch(images, 0{hist_arg})');
                            hist_corr_summary = append_hist_corr( ...
                                hist_corr_summary, mode, iter, 'hist_after_spectrum', shine_log);
                    end
                end

                out_dir = fullfile(output_root, sprintf('mode_%d', mode));
                if ~exist(out_dir, 'dir')
                    mkdir(out_dir);
                end
                for k = 1:numel(files)
                    imwrite(uint8(images{{k}}), fullfile(out_dir, files(k).name));
                end
            end

            print_hist_corr_summary(hist_corr_summary);

            function rows = append_hist_corr(rows, mode, iter, stage, shine_log)
                tokens = regexp( ...
                    shine_log, ...
                    'Correlation between processed image and target histogram:\\s*([0-9.eE+-]+)', ...
                    'tokens');
                if isempty(tokens)
                    return;
                end
                values = zeros(1, numel(tokens));
                for ti = 1:numel(tokens)
                    values(ti) = str2double(tokens{{ti}}{{1}});
                end
                rows(end + 1, :) = {{ ...
                    mode, iter, stage, numel(values), min(values), mean(values), max(values) ...
                }};
            end

            function print_hist_corr_summary(rows)
                if isempty(rows)
                    return;
                end
                fprintf('\\nMATLAB SHINE histMatch correlation summary\\n');
                fprintf('------+-----------+----------------------+--------+-----------+-----------+-----------\\n');
                fprintf('mode  | iteration | stage                | images | min_corr  | mean_corr | max_corr\\n');
                fprintf('------+-----------+----------------------+--------+-----------+-----------+-----------\\n');
                for ri = 1:size(rows, 1)
                    fprintf( ...
                        '%-5d | %-9d | %-20s | %-6d | %.3e | %.3e | %.3e\\n', ...
                        rows{{ri, 1}}, rows{{ri, 2}}, rows{{ri, 3}}, rows{{ri, 4}}, ...
                        rows{{ri, 5}}, rows{{ri, 6}}, rows{{ri, 7}});
                end
                fprintf('------+-----------+----------------------+--------+-----------+-----------+-----------\\n');
            end
            """
        ).strip()
        + "\n"
    )
    return runner


def _build_options(
    *,
    input_dir: Path,
    output_dir: Path,
    mode: int,
    iterations: int,
    seed: int,
    target_hist: np.ndarray | None = None,
    target_spectrum: np.ndarray | None = None,
    legacy_mode: bool = True,
) -> Options:
    return Options(
        input_folder=input_dir,
        output_folder=output_dir,
        mode=mode,
        legacy_mode=legacy_mode,
        as_gray=True,
        # Keep False here: True uses equal-channel grayscale; False uses the
        # rec601 color-treatment path (legacy rgb2gray or modern xyY/Y).
        linear_luminance=False,
        rec_standard=1,
        iterations=iterations,
        seed=seed,
        target_hist=target_hist,
        target_spectrum=target_spectrum,
        verbose=-1,
    )


def compute_python_targets(
    *,
    input_dir: Path,
    output_root: Path,
    seed: int,
    legacy_mode: bool = True,
    target_name: str = "legacy",
    export_matlab_targets: bool = True,
) -> TargetSet:
    target_dir = output_root / f"python_target_source_{target_name}"
    target_dir.mkdir(parents=True, exist_ok=True)
    # Only the initial targets are read from this run; they are computed before
    # the iteration loop, so a single iteration is enough.
    options = _build_options(
        input_dir=input_dir,
        output_dir=target_dir,
        mode=8,
        iterations=1,
        seed=seed,
        legacy_mode=legacy_mode,
    )
    proc = ImageProcessor(dataset=ImageDataset(options=options), options=options)

    target_hist = np.asarray(proc._initial_targets["hist"], dtype=np.float64)
    target_hist_1d = target_hist[:, 0] if target_hist.ndim == 2 else target_hist
    first_image = read_saved_gray(sorted(input_dir.glob("*.png"))[0])
    n_pixels = int(first_image.size)
    target_hist_freq = rounded_target_hist(target_hist_1d, n_pixels)
    target_hist_counts = np.rint(target_hist_freq * n_pixels).astype(np.int64)

    target_spectrum = np.asarray(proc._initial_targets["spectrum"], dtype=np.float64).squeeze()
    radius = get_radius_grid(
        target_spectrum.shape[0],
        target_spectrum.shape[1],
        legacy_mode=legacy_mode,
    )
    target_sf = rotational_avg(target_spectrum, radius)

    if export_matlab_targets:
        target_path = output_root / "python_targets.mat"
        savemat(
            target_path,
            {
                "target_hist_freq": target_hist_freq[:, None],
                "target_hist_counts": target_hist_counts[:, None],
                "target_spectrum": target_spectrum,
                "target_sf": target_sf[:, None],
                "n_pixels": np.array([[n_pixels]], dtype=np.int64),
            },
        )
        (output_root / "python_target_reference.txt").write_text(TARGET_REFERENCE + "\n")
    return {
        "target_name": target_name,
        "legacy_mode": legacy_mode,
        "target_hist_freq": target_hist_freq,
        "target_hist_counts": target_hist_counts,
        "target_spectrum": target_spectrum,
        "target_sf": target_sf,
        "n_pixels": n_pixels,
    }


def run_matlab(matlab_bin: Path, runner: Path) -> None:
    cmd = [str(matlab_bin), "-batch", f"run('{quote_matlab_path(runner)}')"]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        print(
            f"MATLAB exited with status {completed.returncode}; "
            "continuing so output checks can decide.",
            file=sys.stderr,
        )


def run_python_processor(
    *,
    input_dir: Path,
    output_root: Path,
    implementation: str,
    mode: int,
    iterations: int,
    seed: int,
    target_hist: np.ndarray | None = None,
    target_spectrum: np.ndarray | None = None,
    keep_internal: bool = False,
) -> tuple[Path, ImageProcessor]:
    out_dir = output_root / "python" / implementation / f"mode_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    options = _build_options(
        input_dir=input_dir,
        output_dir=out_dir,
        mode=mode,
        iterations=iterations,
        seed=seed,
        target_hist=target_hist,
        target_spectrum=target_spectrum,
        legacy_mode=PYTHON_IMPLEMENTATIONS[implementation][1],
    )
    processor = ImageProcessor(
        dataset=ImageDataset(options=options),
        options=options,
        from_unit_test=keep_internal,
    )
    if keep_internal:
        processor.process()
        processor.print_log_results()
        if not getattr(processor.dataset.images, "has_list_array", False):
            processor.dataset.save_images()
    return out_dir, processor


def read_saved_gray(path: Path, implementation: str = MATLAB_IMPLEMENTATION) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        return image.astype(np.float64)

    rgb = image[..., :3]
    if implementation in {MATLAB_IMPLEMENTATION, LEGACY_IMPLEMENTATION}:
        return MatlabOperators.uint8(
            rgb2gray(rgb, weighting_standard="rec601", matlab_601=True)
        ).astype(np.float64)

    if implementation == MODERN_GRAY_IMPLEMENTATION:
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            return rgb[..., 0].astype(np.float64)
        return np.rint(
            np.clip(rgb2gray(rgb, weighting_standard="rec601", matlab_601=False), 0, 255)
        ).astype(np.float64)

    raise ValueError(f"Unknown saved-output grayscale conversion for: {implementation}")


def read_saved_target_image(path: Path, implementation: str = MATLAB_IMPLEMENTATION) -> np.ndarray:
    if implementation != MODERN_GRAY_IMPLEMENTATION:
        return read_saved_gray(path, implementation=implementation)

    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.dstack([image, image, image])
    rgb = image[..., :3].astype(np.float64) / 255.0
    converter = ColorConverter(rec_standard="rec601")
    return converter.sRGB_to_xyY(rgb)[..., 2] * 255.0


def load_shinier_initial_buffer(
    *,
    input_dir: Path,
    output_root: Path,
    seed: int,
    full_tracking: bool,
) -> dict[str, np.ndarray]:
    probe_dir = output_root / "python" / "internal_input_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    options = _build_options(
        input_dir=input_dir,
        output_dir=probe_dir,
        mode=1,
        iterations=1,
        seed=seed,
    )
    processor = ImageProcessor(dataset=ImageDataset(options=options), options=options)
    input_paths = sorted(input_dir.glob("*.png"))
    internal = {
        path.name: np.asarray(processor._initial_buffer[idx], dtype=np.float64)
        for idx, path in enumerate(input_paths)
    }
    processor.dataset.close()

    if full_tracking:
        out_dir = output_root / "python" / "internal_input_gray"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, image in internal.items():
            Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8)).save(out_dir / name)
    return internal


def hist_l1(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance between two image histograms (= 2 x total variation distance)."""
    ha = np.bincount(a.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    hb = np.bincount(b.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    ha /= max(ha.sum(), 1.0)
    hb /= max(hb.sum(), 1.0)
    return 2.0 * compute_tvd_hist(ha, hb)


def hist_l1_to_target(image: np.ndarray, target_hist_freq: np.ndarray) -> float:
    """L1 distance between an image histogram and a target distribution."""
    binned = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    hist = np.bincount(binned.ravel(), minlength=256).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    target = np.asarray(target_hist_freq, dtype=np.float64).reshape(-1)
    target = target / target.sum()
    return 2.0 * compute_tvd_hist(hist, target)


def exported_hist_target(
    target_hist_freq: np.ndarray,
    *,
    implementation: str,
    n_pixels: int | None = None,
) -> np.ndarray:
    target = np.asarray(target_hist_freq, dtype=np.float64).reshape(-1)
    target = target / target.sum()
    if n_pixels is not None:
        target = rounded_target_hist(target, int(n_pixels))

    if implementation != MODERN_GRAY_IMPLEMENTATION:
        return target

    levels = np.arange(256, dtype=np.float64)
    converter = ColorConverter(rec_standard="rec601")
    srgb = converter.linRGB_to_sRGB(
        np.dstack([levels / 255, levels / 255, levels / 255])
    )[..., 0] * 255
    saved = uint8_plus(srgb)
    target_domain = converter.sRGB_to_xyY(
        np.dstack([saved, saved, saved]).astype(np.float64) / 255
    )[..., 2] * 255
    mapped_bins = np.clip(np.rint(target_domain), 0, 255).astype(np.uint8).ravel()

    mapped = np.zeros_like(target)
    for src_bin, dst_bin in enumerate(mapped_bins):
        mapped[int(dst_bin)] += target[src_bin]
    mapped /= mapped.sum()
    return mapped


def spectrum_rmse_to_target(image: np.ndarray, target_spectrum: np.ndarray) -> float:
    """RMSE between an image's Fourier magnitude spectrum and a target spectrum."""
    magnitude, _ = image_spectrum(image / 255.0, rescale=False)
    observed = magnitude.squeeze().astype(np.float64)
    target = np.asarray(target_spectrum, dtype=np.float64).squeeze()
    return float(compute_rmse(observed, target))


def pre_range_spectrum_metrics(
    *,
    processor: ImageProcessor,
    input_dir: Path,
    mode: int,
    targets: TargetSet,
) -> dict[str, dict[str, float | str]]:
    if mode not in SPECTRUM_MODES:
        return {}

    target = np.asarray(targets["target_spectrum"], dtype=np.float64)
    if target.ndim == 2:
        target = target[..., None]

    rows: dict[str, dict[str, float | str]] = {}
    for idx, input_path in enumerate(sorted(input_dir.glob("*.png"))):
        phase = np.asarray(processor.dataset.phases[idx], dtype=np.float64)
        if phase.ndim == 2:
            phase = phase[..., None]

        original_shape = np.asarray(processor._initial_buffer[idx]).shape[:2]
        pre_range_channels: list[np.ndarray] = []
        fourier_rmses: list[float] = []
        for channel in range(target.shape[-1]):
            xx, yy = pol2cart(target[:, :, channel], phase[:, :, channel])
            reconstructed = np.real(np.fft.ifft2(np.fft.ifftshift(xx + yy * 1j)))
            reconstructed_mag = np.abs(np.fft.fftshift(np.fft.fft2(reconstructed)))
            fourier_rmses.append(float(compute_rmse(reconstructed_mag, target[:, :, channel])))
            pre_range_channels.append(_crop_after_fft(reconstructed, original_shape))

        pre_range01 = np.stack(pre_range_channels, axis=-1).squeeze()
        pre_range255 = pre_range01 * 255.0
        try:
            pre_range_spec_rmse: float | str = spectrum_rmse_to_target(pre_range255, target)
        except ValueError:
            pre_range_spec_rmse = ""

        if pre_range01.min() < 0.0 or pre_range01.max() > 1.0:
            post_clip01 = soft_clip(
                pre_range01,
                min_value=0.0,
                max_value=1.0,
                max_percent=0.01,
                verbose=False,
            )
        else:
            post_clip01 = pre_range01
        post_clip_spec_rmse = spectrum_rmse_to_target(post_clip01 * 255.0, target)

        rows[input_path.name] = {
            "pre_range_fourier_spectrum_rmse_to_python_target": float(np.mean(fourier_rmses)),
            "pre_range_image_spectrum_rmse_to_python_target": pre_range_spec_rmse,
            "post_soft_clip_spectrum_rmse_to_python_target": post_clip_spec_rmse,
            "pre_range_below_zero_fraction": float(np.mean(pre_range01 < 0.0)),
            "pre_range_above_one_fraction": float(np.mean(pre_range01 > 1.0)),
            "pre_range_out_of_range_fraction": float(
                np.mean((pre_range01 < 0.0) | (pre_range01 > 1.0))
            ),
        }
    return rows


def sf_rmse_to_target(image: np.ndarray, target_sf: np.ndarray, legacy_mode: bool = True) -> float:
    """RMSE between an image's rotational-average SF profile and a target profile."""
    magnitude, _ = image_spectrum(image / 255.0, rescale=False)
    observed_mag = magnitude.squeeze().astype(np.float64)
    radius = get_radius_grid(observed_mag.shape[0], observed_mag.shape[1], legacy_mode=legacy_mode)
    observed_sf = rotational_avg(observed_mag, radius)
    target = np.asarray(target_sf, dtype=np.float64).reshape(-1)
    n = min(observed_sf.size, target.size)
    return float(compute_rmse(observed_sf[:n], target[:n]))


def pixel_difference_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        "rmse": float(compute_rmse(a, b)),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(abs_diff.max()),
        "nonzero_pixels": int(np.count_nonzero(abs_diff)),
        "equal_fraction": float(np.mean(abs_diff == 0)),
        "hist_l1": hist_l1(a, b),
    }


def mean_value(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def group_rows(rows: list[dict[str, object]], *keys: str):
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    return sorted(groups.items())


def summarize_pixel_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": len(rows),
        "mean_rmse": mean_value(rows, "rmse"),
        "mean_mae": mean_value(rows, "mae"),
        "max_abs": float(max(float(row["max_abs"]) for row in rows)),
        "mean_equal_fraction": mean_value(rows, "equal_fraction"),
        "mean_hist_l1": mean_value(rows, "hist_l1"),
    }


def target_row(
    *,
    mode: int,
    implementation: str,
    targets: TargetSet,
    image_name: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "mode": mode,
        "implementation": implementation,
        "target": str(targets.get("target_name", "legacy")),
        "image": image_name,
        **{key: "" for key in TARGET_METRIC_KEYS},
        **metrics,
    }


def target_metrics(
    image: np.ndarray,
    *,
    mode: int,
    targets: TargetSet,
    prefix: str,
    hist_target_freq: np.ndarray | None = None,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if mode in HIST_MODES:
        metrics[f"{prefix}_hist_l1_to_python_target"] = hist_l1_to_target(
            image,
            hist_target_freq if hist_target_freq is not None else targets["target_hist_freq"],
        )
    if mode in SF_MODES:
        metrics[f"{prefix}_sf_rmse_to_python_target"] = sf_rmse_to_target(
            image,
            targets["target_sf"],
            legacy_mode=bool(targets.get("legacy_mode", True)),
        )
    if mode in SPECTRUM_MODES:
        metrics[f"{prefix}_spectrum_rmse_to_python_target"] = spectrum_rmse_to_target(
            image, targets["target_spectrum"]
        )
    return metrics


def exported_target_metrics(
    image: np.ndarray,
    *,
    implementation: str,
    mode: int,
    targets: TargetSet,
) -> dict[str, object]:
    hist_target_freq = None
    if mode in HIST_MODES:
        hist_target_freq = exported_hist_target(
            targets["target_hist_freq"],
            implementation=implementation,
            n_pixels=int(targets["n_pixels"]),
        )
    return target_metrics(
        image,
        mode=mode,
        targets=targets,
        prefix="exported_png",
        hist_target_freq=hist_target_freq,
    )


def compare_dirs(
    matlab_dir: Path,
    python_dir: Path,
    mode: int,
    implementation: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for matlab_path in sorted(matlab_dir.glob("*.png")):
        python_path = python_dir / matlab_path.name
        if not python_path.exists():
            raise FileNotFoundError(f"Missing Python output: {python_path}")
        m = read_saved_gray(matlab_path, implementation=MATLAB_IMPLEMENTATION)
        p = read_saved_gray(python_path, implementation=implementation)
        rows.append({
            "mode": mode,
            "reference": MATLAB_IMPLEMENTATION,
            "python_output": implementation,
            "image": matlab_path.name,
            **pixel_difference_metrics(p, m),
        })
    return rows


def compare_input_grayscale(
    input_dir: Path,
    output_root: Path,
    *,
    seed: int,
    full_tracking: bool = False,
) -> list[dict[str, object]]:
    matlab_gray_dir = output_root / "matlab" / "input_gray"
    if not matlab_gray_dir.exists():
        return []
    shinier_internal = load_shinier_initial_buffer(
        input_dir=input_dir,
        output_root=output_root,
        seed=seed,
        full_tracking=full_tracking,
    )

    rows: list[dict[str, object]] = []
    for matlab_path in sorted(matlab_gray_dir.glob("*.png")):
        if matlab_path.name not in shinier_internal:
            raise FileNotFoundError(
                f"Missing SHINIER internal input buffer for {matlab_path.name}"
            )
        m = read_saved_gray(matlab_path, implementation=MATLAB_IMPLEMENTATION)
        p = shinier_internal[matlab_path.name]
        rows.append({"image": matlab_path.name, **pixel_difference_metrics(p, m)})
    write_csv(output_root / "input_grayscale_comparison.csv", rows)
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for (mode, reference, python_output), items in group_rows(
        rows, "mode", "reference", "python_output"
    ):
        summary = summarize_pixel_rows(items)
        out.append(
            {
                "mode": int(mode),
                "reference": reference,
                "python_output": python_output,
                **{key: summary[key] for key in PIXEL_COLUMNS if key != "images"},
            }
        )
    return out


def compare_outputs_to_python_targets(
    *,
    output_dir: Path,
    implementation: str,
    mode: int,
    targets: TargetSet,
) -> list[dict[str, object]]:
    if mode not in TARGET_MODES:
        return []
    rows: list[dict[str, object]] = []
    for image_path in sorted(output_dir.glob("*.png")):
        image = read_saved_target_image(image_path, implementation=implementation)
        rows.append(
            target_row(
                mode=mode,
                implementation=implementation,
                targets=targets,
                image_name=image_path.name,
                metrics=exported_target_metrics(
                    image,
                    implementation=implementation,
                    mode=mode,
                    targets=targets,
                ),
            )
        )
    return rows


def compare_processor_to_python_targets(
    *,
    processor: ImageProcessor,
    input_dir: Path,
    output_dir: Path,
    implementation: str,
    mode: int,
    targets: TargetSet,
) -> list[dict[str, object]]:
    if mode not in TARGET_MODES:
        return []
    rows: list[dict[str, object]] = []
    pre_range_by_image = pre_range_spectrum_metrics(
        processor=processor,
        input_dir=input_dir,
        mode=mode,
        targets=targets,
    )
    for idx, input_path in enumerate(sorted(input_dir.glob("*.png"))):
        image = np.asarray(processor._final_buffer[idx]).squeeze().astype(np.float64)
        metrics = {
            **target_metrics(image, mode=mode, targets=targets, prefix="internal_buffer"),
            **pre_range_by_image.get(input_path.name, {}),
        }

        saved_path = output_dir / input_path.name
        if saved_path.exists():
            exported_image = read_saved_target_image(saved_path, implementation=implementation)
            metrics.update(
                exported_target_metrics(
                    exported_image,
                    implementation=implementation,
                    mode=mode,
                    targets=targets,
                )
            )
        rows.append(
            target_row(
                mode=mode,
                implementation=implementation,
                targets=targets,
                image_name=input_path.name,
                metrics=metrics,
            )
        )
    return rows


def aggregate_target_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for (mode, implementation, target), items in group_rows(
        rows, "mode", "implementation", "target"
    ):
        summary: dict[str, object] = {
            "mode": int(mode),
            "implementation": implementation,
            "target": target,
        }
        for key in TARGET_METRIC_KEYS:
            vals = [float(item[key]) for item in items if item[key] != ""]
            summary[f"mean_{key}"] = float(np.mean(vals)) if vals else ""
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cleanup_tracking_artifacts(run_root: Path) -> None:
    if not run_root.exists():
        return
    for path in run_root.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        elif path.suffix.lower() != ".csv":
            path.unlink()


INTEGER_COLUMNS = {"mode", "images", "nonzero_pixels"}


def format_table_value(column: str, value: object) -> str:
    if value == "":
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if column in INTEGER_COLUMNS and isinstance(value, (int, float, np.integer, np.floating)):
        return str(int(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.3e}"
    return str(value)


def print_table(
    title: str,
    rows: list[dict[str, object]],
    columns: list[str],
    *,
    column_labels: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> None:
    if not rows:
        return
    labels = column_labels or {}
    headers = [labels.get(column, column) for column in columns]
    formatted_rows = [
        [format_table_value(column, row.get(column, "")) for column in columns]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in formatted_rows))
        for index, column in enumerate(columns)
    ]
    header = " | ".join(headers[index].ljust(widths[index]) for index, _ in enumerate(columns))
    separator = "-+-".join("-" * width for width in widths)

    print()
    print(title)
    print(separator)
    print(header)
    print(separator)
    for row in formatted_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print(separator)
    if notes:
        for note in notes:
            print(note)


def print_key_values(title: str, row: dict[str, object], keys: list[str]) -> None:
    key_width = max(len(key) for key in keys)

    print()
    print(title)
    print("-" * len(title))
    for key in keys:
        value = row.get(key, "")
        if isinstance(value, bool):
            text = str(value).lower()
        else:
            text = str(value)
        print(f"{key.ljust(key_width)} : {text}")


def prepare_run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_root = (
        args.run_root or args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    input_dir = run_root / "input"
    if not input_dir.exists() or not list(input_dir.glob("*.png")):
        copied_inputs = prepare_inputs(args.input_dir, run_root, args.limit)
        input_dir = copied_inputs[0].parent

    runner = write_matlab_runner(
        output_root=run_root,
        input_dir=input_dir,
        shine_dir=args.shine_dir,
        modes=args.modes,
        iterations=args.iterations,
        seed=args.seed,
        use_python_targets=args.use_python_targets,
    )
    if not args.quiet_run_info:
        print_key_values(
            "Prepared diagnostic run",
            {
                "run_root": run_root,
                "matlab_runner": runner,
                "modes": " ".join(str(mode) for mode in args.modes),
                "images": len(list(input_dir.glob("*.png"))),
                "use_python_targets": args.use_python_targets,
                "skip_matlab": args.skip_matlab,
                "prepare_only": args.prepare_only,
            },
            RUN_INFO_KEYS,
        )
    return run_root, input_dir, runner


def compute_target_sets(
    *,
    input_dir: Path,
    run_root: Path,
    seed: int,
) -> TargetSets:
    configs = [
        ("legacy", True, True),
        ("modern_gray", False, False),
    ]
    return {
        name: compute_python_targets(
            input_dir=input_dir,
            output_root=run_root,
            seed=seed,
            legacy_mode=legacy_mode,
            target_name=name,
            export_matlab_targets=export_matlab_targets,
        )
        for name, legacy_mode, export_matlab_targets in configs
    }


def print_input_grayscale_results(rows: list[dict[str, object]], run_root: Path) -> None:
    if rows:
        print(f"input_gray_summary={run_root / 'input_grayscale_comparison.csv'}")
        print_table(INPUT_GRAY_TITLE, [summarize_pixel_rows(rows)], PIXEL_COLUMNS)
        return
    print_table(
        INPUT_GRAY_TITLE,
        [
            {
                "status": "not_available",
                "reason": "MATLAB input_gray outputs were not found",
            }
        ],
        ["status", "reason"],
    )


def matlab_target_rows(
    *,
    run_root: Path,
    mode: int,
    skip_matlab: bool,
    target_sets: TargetSets,
) -> list[dict[str, object]]:
    matlab_dir = run_root / "matlab" / f"mode_{mode}"
    if skip_matlab and not matlab_dir.exists():
        return []
    if not matlab_dir.exists():
        raise FileNotFoundError(f"Missing MATLAB outputs for mode {mode}: {matlab_dir}")
    return compare_outputs_to_python_targets(
        output_dir=matlab_dir,
        implementation=MATLAB_IMPLEMENTATION,
        mode=mode,
        targets=target_sets["legacy"],
    )


def python_target_rows(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    run_root: Path,
    mode: int,
    target_sets: TargetSets,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for implementation in PYTHON_IMPLEMENTATIONS:
        targets = target_sets[PYTHON_IMPLEMENTATIONS[implementation][0]]
        python_dir, processor = run_python_processor(
            input_dir=input_dir,
            output_root=run_root,
            implementation=implementation,
            mode=mode,
            iterations=args.iterations,
            seed=args.seed,
            target_hist=np.asarray(targets["target_hist_freq"], dtype=np.float64),
            target_spectrum=np.asarray(targets["target_spectrum"], dtype=np.float64),
            keep_internal=True,
        )
        try:
            rows.extend(
                compare_processor_to_python_targets(
                    processor=processor,
                    input_dir=input_dir,
                    output_dir=python_dir,
                    implementation=implementation,
                    mode=mode,
                    targets=targets,
                )
            )
        finally:
            processor.dataset.close()
    return rows


def run_target_probe(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    run_root: Path,
    target_sets: TargetSets,
) -> None:
    target_detail_rows: list[dict[str, object]] = []
    for mode in args.modes:
        target_detail_rows.extend(
            matlab_target_rows(
                run_root=run_root,
                mode=mode,
                skip_matlab=args.skip_matlab,
                target_sets=target_sets,
            )
        )
        target_detail_rows.extend(
            python_target_rows(
                args=args,
                input_dir=input_dir,
                run_root=run_root,
                mode=mode,
                target_sets=target_sets,
            )
        )

    target_summary_rows = aggregate_target_rows(target_detail_rows)
    write_csv(run_root / "python_target_probe_detail.csv", target_detail_rows)
    write_csv(run_root / "python_target_probe_summary.csv", target_summary_rows)
    print(f"summary={run_root / 'python_target_probe_summary.csv'}")
    print_table(
        "MATLAB/Python vs implementation-specific fixed initial Python target",
        target_summary_rows,
        TARGET_TABLE_COLUMNS,
        column_labels=TARGET_TABLE_LABELS,
        notes=TARGET_TABLE_NOTES,
    )


def run_matlab_vs_python_probe(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    run_root: Path,
) -> None:
    detail_rows: list[dict[str, object]] = []
    for mode in args.modes:
        matlab_dir = run_root / "matlab" / f"mode_{mode}"
        if not matlab_dir.exists():
            raise FileNotFoundError(f"Missing MATLAB outputs for mode {mode}: {matlab_dir}")
        for implementation in PYTHON_IMPLEMENTATIONS:
            python_dir, processor = run_python_processor(
                input_dir=input_dir,
                output_root=run_root,
                implementation=implementation,
                mode=mode,
                iterations=args.iterations,
                seed=args.seed,
            )
            processor.dataset.close()
            detail_rows.extend(compare_dirs(matlab_dir, python_dir, mode, implementation))

    summary_rows = aggregate(detail_rows)
    write_csv(run_root / "matlab_shine_comparison_detail.csv", detail_rows)
    write_csv(run_root / "matlab_shine_comparison_summary.csv", summary_rows)
    print(f"summary={run_root / 'matlab_shine_comparison_summary.csv'}")
    print_table(
        "Python outputs relative to MATLAB SHINE",
        summary_rows,
        OUTPUT_COMPARISON_COLUMNS,
    )


def main() -> None:
    args = parse_args()
    run_root, input_dir, runner = prepare_run(args)

    target_sets = None
    if args.use_python_targets:
        print(ANSI_BOLD_RED + "\n".join(f"WARNING: {line}" for line in TARGET_WARNING_LINES) + ANSI_RESET)
        target_sets = compute_target_sets(
            input_dir=input_dir,
            run_root=run_root,
            seed=args.seed,
        )
    if args.prepare_only:
        return
    if not args.skip_matlab:
        run_matlab(args.matlab_bin, runner)

    input_gray_rows = compare_input_grayscale(
        input_dir,
        run_root,
        seed=args.seed,
        full_tracking=args.full_tracking,
    )
    print_input_grayscale_results(input_gray_rows, run_root)

    if args.use_python_targets:
        assert target_sets is not None
        run_target_probe(
            args=args,
            input_dir=input_dir,
            run_root=run_root,
            target_sets=target_sets,
        )
    else:
        run_matlab_vs_python_probe(args=args, input_dir=input_dir, run_root=run_root)

    if not args.full_tracking:
        cleanup_tracking_artifacts(run_root)


if __name__ == "__main__":
    main()
