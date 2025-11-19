#!/usr/bin/env bash
# Test `pip install .` and `pip install ".[dev]"` across multiple Python versions
# using version-specific virtualenvs on the local machine.
#
# Usage (from anywhere inside the repo):
#   bash tests/other_tests/test_install_all_pythons.sh
#
# What it does
# ------------
# - Loops over Python 3.MIN_MINOR .. 3.MAX_MINOR.
# - For each python3.X found in PATH:
#     * Creates a venv at .venv-pyX (e.g., .venv-py9, .venv-py10, ...)
#     * Activates it
#     * Runs:
#         - python -m pip install .
#         - python -m pip install ".[dev]"
# - Prints a summary of:
#     * Versions where BOTH installs SUCCEEDED
#     * Versions where either install FAILED
#     * Versions that were MISSING (interpreter not found)
#
# Installing missing Python versions
# ----------------------------------
# On macOS with Homebrew:
#   brew install python@3.9 python@3.10 python@3.11 python@3.12
#
# This will give you python3.9, python3.10, python3.11, python3.12 in PATH
# (after `brew doctor` / `brew info` instructions, or re-opening your shell).
#
# On Linux:
#   - Use your distro package manager (apt, dnf, pacman, etc.) or `pyenv`.
#
# On Windows:
#   - Use the official installers from python.org or pyenv-win, then ensure
#     `python3.X` is on PATH (or adapt this script to use pyenv shims).

set -u  # treat unset variables as errors
set -o pipefail

# --- CONFIG -------------------------------------------------------------

# Min and max Python 3 minor versions to try (inclusive).
MIN_MINOR=9          # Python 3.9
MAX_MINOR=14         # Python 3.14 (upper bound in pyproject requires-python)

# Resolve project root (where pyproject.toml / setup.py lives).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# -----------------------------------------------------------------------

ok_versions=()
fail_versions=()
missing_versions=()

cd "$PROJECT_DIR" || {
  echo "[ERROR] Cannot cd into project dir: $PROJECT_DIR"
  exit 1
}

if [[ ! -f "pyproject.toml" && ! -f "setup.py" ]]; then
  echo "[ERROR] No pyproject.toml or setup.py in $PROJECT_DIR"
  exit 1
fi

echo "Testing pip install . and pip install '.[dev]' from Python 3.${MIN_MINOR} to 3.${MAX_MINOR}"
echo "Project directory: $PROJECT_DIR"
echo

for minor in $(seq "$MIN_MINOR" "$MAX_MINOR"); do
  pybin="python3.${minor}"

  if ! command -v "$pybin" >/dev/null 2>&1; then
    echo "[SKIP] $pybin not found in PATH"
    missing_versions+=("3.${minor}")
    continue
  fi

  venv_dir="$PROJECT_DIR/.venv-py${minor}"
  echo
  echo "=== Python 3.${minor} → $pybin ==="
  echo "[INFO] Creating venv at: $venv_dir"

  # Clean any previous venv for this version
  rm -rf "$venv_dir"

  if ! "$pybin" -m venv "$venv_dir"; then
    echo "[FAIL] Could not create venv with $pybin"
    fail_versions+=("3.${minor} (venv creation failed)")
    continue
  fi

  # Activate the venv (this sets THIS SHELL's Python to the right version)
  # shellcheck disable=SC1090
  source "$venv_dir/bin/activate"

  echo "[INFO] Using Python: $(python -V 2>&1)"
  echo "[INFO] Upgrading pip..."
  if ! python -m pip install --upgrade pip >/dev/null 2>&1; then
    echo "[WARN] pip upgrade failed for Python 3.${minor} (continuing anyway)"
  fi

  # 1) Base install
  echo "[INFO] Running: pip install ."
  if python -m pip install .; then
    echo "[OK] Base install succeeded with Python 3.${minor}"

    # 2) Dev extras install
    echo "[INFO] Running: pip install '.[dev]'"
    if python -m pip install ".[dev]"; then
      echo "[OK] Dev extras install succeeded with Python 3.${minor}"
      ok_versions+=("3.${minor}")
    else
      echo "[FAIL] Dev extras install FAILED with Python 3.${minor}"
      fail_versions+=("3.${minor} (dev extras)")
    fi
  else
    echo "[FAIL] Base install FAILED with Python 3.${minor}"
    fail_versions+=("3.${minor} (base)")
  fi

  # Deactivate venv before next loop
  deactivate || true
done

echo
echo "================ SUMMARY ================"
if ((${#ok_versions[@]} > 0)); then
  echo "Success (base + dev extras):"
  for v in "${ok_versions[@]}"; do
    echo "  - Python $v"
  done
else
  echo "Success: none"
fi

echo
if ((${#fail_versions[@]} > 0)); then
  echo "Failures:"
  for v in "${fail_versions[@]}"; do
    echo "  - Python $v"
  done
else
  echo "Failures: none"
fi

echo
if ((${#missing_versions[@]} > 0)); then
  echo "Missing interpreters (not found in PATH):"
  for v in "${missing_versions[@]}"; do
    echo "  - Python $v"
  done
  echo
  echo "Hints:"
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    echo "  On macOS with Homebrew, you can install them with:"
    echo -n "    brew install"
    for v in "${missing_versions[@]}"; do
      # strip '3.' and build python@3.X formula name
      minor="${v#3.}"
      echo -n " python@3.${minor}"
    done
    echo
  else
    echo "  Install the missing Python 3 versions using your OS package manager"
    echo "  or pyenv, then re-run this script."
  fi
else
  echo "Missing interpreters: none"
fi
echo "========================================="

echo
echo "IMPORTANT: .venv-pyX directories are left on disk for inspection."
echo "           You can safely delete them when done (rm -rf .venv-py*)."