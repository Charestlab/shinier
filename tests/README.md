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
| `test_all_options` | Exhaustive unit tests on Options (slow) | `pytest -m tests_all_options` |

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

## ⚡ Validation Tests with Shards

Shards divide the exhaustive test space across multiple processes.

### Environment variables

| Variable           | Description                                | Default |
| ------------------ | ------------------------------------------ | ------- |
| `SHARDS`           | Total number of shards                     | `1`     |
| `SHARD_INDEX`      | Index of current shard (0-based)           | `0`     |
| `SHOW_PROGRESS`    | Enable tqdm progress bars                  | `0`     |
| `DUMP_FILE_FORMAT` | Format for failure dumps (`json` or `pkl`) | `json`  |
| `START_AT`         | Resume testing from given combo index      | `0`     |

---

## 🛠️ Running Shards in Parallel

Use **GNU parallel** to distribute shards across CPU cores:

```bash
parallel --ungroup --jobs 8 \
  'PYTHONUNBUFFERED=1 SHOW_PROGRESS=1 DUMP_FILE_FORMAT=pkl SHARDS=8 SHARD_INDEX={} pytest -s -m validation_tests' ::: 0 1 2 3 4 5 6 7
```

> 🔹 `--ungroup` allows live tqdm updates in real time.\
> Without it, each shard's output is buffered until completion.

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

## 🔄 Resume From a Given Combo

If a bug occurs at combo 21,600 (from tqdm output):

```bash
START_AT=21600 DUMP_FILE_FORMAT=pkl SHOW_PROGRESS=1 SHARDS=8 SHARD_INDEX=0 \
pytest -m validation_tests -vv -s --maxfail=1 --pdb
```

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

1. Run exhaustive validation tests across 8 shards:

   ```bash
   parallel --ungroup --jobs 8 'SHOW_PROGRESS=1 DUMP_FILE_FORMAT=pkl SHARDS=8 SHARD_INDEX={} pytest -m validation_tests -x -s' ::: 0 1 2 3 4 5 6 7
   ```

2. Inspect failures:

   ```bash
   ls tests/assets/tmp/**/failure_*.pkl
   ```

3. Replay a failure interactively:

   ```bash
   python -m tests.tools.replay_failure path/to/failure_xxxxx.pkl
   ```

