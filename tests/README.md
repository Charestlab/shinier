# Testing Guide

This repository uses **pytest** for all testing.\
Tests are organized into **unit** and **validation (integration)** levels.

---

## ⚙️ Pytest Configuration
Make sure you did to install all dev dependencies: ```pip install '.[dev]'```

```ini
[pytest]
minversion = 7.0
addopts = -ra -q
testpaths =
    tests/unit_tests
    tests/validation_tests
markers =
    unit_tests: fast functional unit tests
    validation_tests: full combinatorial sweep (slow)
    test_all_options: test all option combinations (slow)
```

**Naming convention:**\
All test files must either:

- start with `test_`, or
- end with `_test.py`\
  (e.g., `ImageDataset_test.py` ✅)

---

## 📙 Markers

Use markers to select subsets of tests:

| Marker             | Description                             | Example Command               |
|--------------------|-----------------------------------------|-------------------------------|
| `unit_tests`       | Fast functional unit tests              | `pytest -m unit_tests`        |
| `validation_tests` | Exhaustive validation (slow)            | `pytest -m validation_tests`  |
| `test_all_options` | Exhaustive unit tests on Options (⚠ very slow; can take a few hours) | `pytest -m test_all_options` |

--- 

## 🧵 Multi-Core Execution

Run tests in parallel automatically:

```bash
pytest -n auto -s -m unit_tests
```

Or specify the number of cores explicitly:

```bash
pytest -n 4 -s -m unit_tests
```

---

## ⚡ Validation Tests — Coverage Modes

`ImageProcessor_validation_test.py` supports three coverage modes selected via `COVERAGE_MODE`:

| Mode          | Description                                                   | Typical shards |
|---------------|---------------------------------------------------------------|----------------|
| `sampled`     | Random sample of the full parameter space (default)           | 1–10           |
| `pruned`      | Full Cartesian product over a reduced parameter set           | 10–20          |
| `exhaustive`  | Full Cartesian product over every possible combination        | 100+           |

**Pruned mode** collapses parameters whose values share the same code path (e.g. all `ie_methods` go through the same dispatch, `rec_standard` values differ only in RGB→gray weights).
Every collapsed value is independently covered by `tests/unit_tests/PrunedCoverage_test.py`, which verifies that each excluded value runs without error and produces finite output — making pruned-mode results safely extrapolatable.

### Environment variables

| Variable           | Description                                              | Default   |
| ------------------ | -------------------------------------------------------- | --------- |
| `COVERAGE_MODE`    | `sampled` / `pruned` / `exhaustive`                      | `sampled` |
| `SHARDS`           | Total number of shards                                   | `1`       |
| `SHARD_INDEX`      | Index of current shard (0-based)                         | `0`       |
| `SHOW_PROGRESS`    | Enable tqdm progress bars                                | `0`       |
| `PERCENT_SAMPLED`  | Fraction of combinations per shard (sampled mode only)   | `0.001`   |
| `DUMP_FILE_FORMAT` | Format for failure dumps (`json` or `pkl`)               | `json`    |
| `START_AT`         | Resume exhaustive/pruned from given combo index          | `0`       |
| `RESTART`          | Re-initialize the hash registry before starting          | `false`   |

Hash keys are namespaced by `COVERAGE_MODE` (e.g. `exhaustive:5` vs `pruned:5`) so separate runs never collide in the shared `tests/hash_registry.db`.

---

## 🛠️ Running Shards Locally (GNU parallel)

Use **GNU parallel** to distribute shards across CPU cores:

```bash
parallel --ungroup --jobs 8 \
  'COVERAGE_MODE=sampled SHARDS=8 SHARD_INDEX={} SHOW_PROGRESS=1 DUMP_FILE_FORMAT=pkl \
   pytest -s -m validation_tests' ::: 0 1 2 3 4 5 6 7
```

> 🔹 `--ungroup` allows live tqdm updates in real time.\
> Without it, each shard's output is buffered until completion.

---

## 🖥️ Running on a Compute Cluster (SLURM/sbatch)

Use a SLURM job array so `$SLURM_ARRAY_TASK_ID` maps directly to `SHARD_INDEX`.
Create a file (e.g. `run_validation.sh`) with the following template and adapt `N_SHARDS`,
`COVERAGE_MODE`, `PERCENT_SAMPLED`, and `--time` to your needs:

```bash
#!/bin/bash
#SBATCH --job-name=shinier_validation
#SBATCH --array=0-<N_SHARDS-1>        # e.g. 0-99 for 100 shards
#SBATCH --time=48:00:00               # 48 h for exhaustive; 12 h for pruned; 2 h for sampled
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/val_%A_%a.out
#SBATCH --error=logs/val_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ....../venv/bin/activate

COVERAGE_MODE=exhaustive \     # exhaustive | pruned | sampled
SHARDS=$SLURM_ARRAY_TASK_COUNT \
SHARD_INDEX=$SLURM_ARRAY_TASK_ID \
START_AT=0 \                   # set > 0 to resume after a crash
SHOW_PROGRESS=0 \
DUMP_FILE_FORMAT=pkl \
python -m pytest tests/validation_tests/ImageProcessor_validation_test.py \
  -m validation_tests -v --tb=short
```

Submit with:
```bash
mkdir -p logs
sbatch run_validation.sh
```

To resume exhaustive mode from combo `1180676` after a crash:
```bash
START_AT=1180676 sbatch run_validation.sh
```

---

## 🔎 Debugging Tests

### Drop into debugger on failure

```bash
pytest -m validation_tests -s --maxfail=1 --pdb
```

### Step interactively inside test

```bash
pytest -m validation_tests -s --trace
```

### Show full traceback

```bash
pytest -m validation_tests -vv -s --tb=long
```

---

## 🟡 Hard vs Soft Failures in Validation Tests

`ImageProcessor_validation_test.py` distinguishes two levels of failure:

| Level | When it fires | Test outcome |
|-------|--------------|--------------|
| **Hard failure** | Unexpected exception or internal validation flag | Test **fails** — something is definitely broken |
| **Soft failure** | Optimization anomaly that is expected in some combos | Test **passes** — issue is printed and dumped, but doesn't block the run |

### Hard failures (call `_dump_and_fail`)
- Any `proc.validation` entry with `valid_result=False` (e.g. histogram RMSE didn't converge to 0 in single-objective modes).
- RMSE more than doubled or increased > 0.1 in absolute terms in composite modes (5–8).
- SSIM `final` check: the actual output Y has lower SSIM than the first optimization iteration — the optimizer went backward overall.

### Soft failures (collected in `combo_warnings`, dumped but don't fail)
- RMSE regressed strictly but by a small amount — expected in composite modes (spec_match undoes histogram progress in mode 6, etc.) or when `hist_optim=True` explicitly trades RMSE for SSIM.
- SSIM `sub_iter` check: the last *pre-rollback proposal* of the SSIM optimizer had lower SSIM than the first — a known artifact of the rollback mechanism (the final output may still be fine).

Soft failures produce a `.pkl` / `.json` dump in `tests/assets/tmp/` for inspection, and are printed as a summary at the end of the test run.

---

## 🔄 Resume From a Given Combo

If a bug occurs at combo 21,600 (from tqdm output):

```bash
START_AT=21600 DUMP_FILE_FORMAT=pkl SHOW_PROGRESS=1 SHARDS=8 SHARD_INDEX=0 \
pytest -m validation_tests -vv -s --maxfail=1 --pdb
```

---

## Image Enhancement MATLAB Reference Hashes

Each image-enhancement algorithm in `ie_methods` is validated pixel-strictly against its free
MATLAB reference implementation.  The test file is
`tests/validation_tests/ImageEnhancement_validation_test.py`.

**Approach:** reference outputs were generated once from
`tests/assets/SAMPLE_512X512/*.png`, converted to raw `uint8` bytes, and stored
as SHA256 hashes in `tests/assets/image_enhancement_matlab_sha256.json`.
The JSON stores the shared `dtype`, `shape`, and ordered `images` list once;
each algorithm stores only the SHA256 list in that same image order. The Python
test recomputes each output and compares shape, dtype, and hash.

Full citations are in the *Implemented algorithms* section of
`documentation/documentation.md`.

| Algorithm | MATLAB fn   | Parameters                  | JSON key  | MATLAB code DOI                    |
|-----------|-------------|-----------------------------|-----------|------------------------------------|
| TIDHE     | `imTIDHE`   | defaults                    | `tidhe`   | 10.13140/RG.2.2.22946.70088        |
| RDFHE     | `imRDFHE`   | `p=10`                      | `rdfhe`   | 10.13140/RG.2.2.23921.34408        |
| NFLDICE   | `imNFLDICE` | `B=10, E_l=5, P_l=127.5`   | `nfldice` | 10.13140/RG.2.2.14716.51849        |
| BETCE     | `imBETCE`   | defaults                    | `betce`   | 10.13140/RG.2.2.14319.09126        |
| SFCEF     | `imSFCEF`   | `t=0.5`                     | `sfcef`   | 10.13140/RG.2.2.16448.44807        |

TIDHE, RDFHE, NFLDICE, and BETCE use exact SHA256 validation against MATLAB
reference hashes.

SFCEF uses a different validation strategy because MATLAB's `imSFCEF` relies on
`filter2`.  MATLAB delegates this convolution to platform libraries that can use
Fused Multiply-Add (FMA) instructions, producing float64 values that differ from
NumPy by a few units in the last place before final `uint8` rounding (see
*MATLAB vs Python Differences §5* in `documentation/documentation.md`).  Because
of this, exact SHA256 equality is not a stable requirement for SFCEF.

The strict SFCEF validation therefore loads MATLAB `imSFCEF` reference images
from `tests/assets/sfcef_matlab_reference/` and asserts that Python
`sfcef_gray(..., legacy_mode=True)` differs by **at most 1 gray level per
pixel** (`max_diff ≤ 1`).  The JSON still stores SFCEF MATLAB SHA256 hashes as a
reference record, but SFCEF's pass/fail test uses the pixel-difference bound
instead of exact hash equality.

We also ran a broader MATLAB-vs-Python comparison on the 500 low-light images of
the LOL dataset
(`https://www.kaggle.com/datasets/soumikrakshit/lol-dataset`).  MATLAB outputs
were generated as `imread -> rgb2gray -> imSFCEF`; Python outputs used
MATLAB-compatible grayscale conversion followed by `sfcef_gray(...,
legacy_mode=True)`.  No pixel differed by more than one gray level, and
0.45075667% of pixels differed by exactly one gray level.  The goal of this
larger comparison was not to claim that one implementation is better, but to
verify that the numerical differences are negligible and do not systematically
favor either implementation.

Mean metric values on LOL were nearly identical: AMBE against the paired LOL
high image was 42.74335064 for Python and 42.74440659 for MATLAB; MSSIM was
0.66327749 for Python and 0.66325624 for MATLAB; PSNR was 14.74127390 for
Python and 14.74096545 for MATLAB; BP2BPSIM was 0.53723447 for Python and
0.53723595 for MATLAB; CI was 58.21891968 for Python and 58.21938888 for
MATLAB; entropy was 6.85142815 for Python and 6.85281768 for MATLAB.  Maximum
absolute metric differences, expressed as percentages of the MATLAB values,
were AMBE 0.007961%, MSSIM 0.019079%, PSNR 0.014468%, BP2BPSIM 0.024168%, CI
0.030034%, and entropy 0.063809%.

---

## 🤍 Replay a Dumped Failure

To reproduce a failed validation test:

```bash
python -m tests.tools.replay_failure /path/to/failure_ab12cd34.pkl
```

This will rebuild the same `Options`, reload selected images, and re-run the failed processing step for debugging (including PyCharm breakpoints).

---

## 🔹 Tips

- Use `--pdb` or `--trace` for interactive debugging.
- Always set `PYTHONUNBUFFERED=1` in `parallel` to force live output.
- Use `DUMP_FILE_FORMAT=pkl` for more reliable replay.
- For CI or remote runs, redirect shard logs:
  ```bash
  parallel --jobs 8 'pytest -m validation_tests -s > shard_{}.log 2>&1' ::: 0 1 2 3 4 5 6 7
  ```
  Then view a specific log with:
  ```bash
  tail -f shard_3.log
  ```

---

## 🔧 Example Workflow

1. Run sampled validation tests locally across 8 cores:

   ```bash
   parallel --ungroup --jobs 8 \
     'COVERAGE_MODE=sampled SHOW_PROGRESS=1 DUMP_FILE_FORMAT=pkl SHARDS=8 SHARD_INDEX={} \
      pytest -m validation_tests -s' ::: 0 1 2 3 4 5 6 7
   ```

2. Inspect failures:

   ```bash
   ls tests/assets/tmp/**/failure_*.pkl
   ```

3. Replay a failure interactively:

   ```bash
   python -m tests.tools.replay_failure path/to/failure_xxxxx.pkl
   ```
