#!/usr/bin/env bash
# Test `pip install .`, `pip install ".[dev]"`, and pytest across multiple OS/Python combos using Docker.

set -u
set -o pipefail

# Resolve project root (where pyproject.toml lives).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -f "$PROJECT_DIR/pyproject.toml" && ! -f "$PROJECT_DIR/setup.py" ]]; then
  echo "[ERROR] No pyproject.toml or setup.py in $PROJECT_DIR"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker is not installed or not in PATH"
  exit 1
fi

# Matrix of base images (OS + Python)
IMAGES=(
  "python:3.9-slim-bookworm"      # Debian (slim)
  "python:3.10-slim-bookworm"
  "python:3.11-slim-bookworm"
  "python:3.12-slim-bookworm"
)

ok=()
fail=()

echo "Project directory: $PROJECT_DIR"
echo "Docker images to test:"
for img in "${IMAGES[@]}"; do
  echo "  - $img"
done
echo

for img in "${IMAGES[@]}"; do
  echo "=== Testing in image: $img ==="

  if ! docker pull "$img" >/dev/null 2>&1; then
    echo "[FAIL] Could not pull image $img"
    fail+=("$img (pull failed)")
    continue
  fi

  # Run container, mount project, install compiler toolchain, create venv, pip installs, pytest
  if docker run --rm \
      -v "$PROJECT_DIR":/project \
      -w /project \
      "$img" \
      /bin/sh -lc "
        set -eu

        echo '[INFO] Base image:'; cat /etc/os-release 2>/dev/null || true

        # --- Install C/C++ toolchain -------------------------------------------------
        if command -v apt-get >/dev/null 2>&1; then
          echo '[INFO] Installing build-essential via apt-get...'
          apt-get update >/dev/null
          apt-get install -y build-essential >/dev/null
          rm -rf /var/lib/apt/lists/*
        elif command -v apk >/dev/null 2>&1; then
          echo '[INFO] Installing build-base via apk...'
          apk add --no-cache build-base >/dev/null
        else
          echo '[WARN] No known package manager (apt-get/apk); assuming compiler is present'
        fi

        echo '[INFO] g++ version (if present):'
        g++ --version || echo 'g++ not found'

        echo '[INFO] Python version:'
        python -V || python3 -V

        echo '[INFO] Creating virtualenv...'
        python -m venv .venv || python3 -m venv .venv

        . .venv/bin/activate

        echo '[INFO] Upgrading pip...'
        python -m pip install --upgrade pip >/dev/null

        echo '[INFO] Installing project (pip install .)...'
        python -m pip install .

        echo '[INFO] Installing project with dev extras (pip install \".[dev]\")...'
        python -m pip install \".[dev]\"

        echo '[INFO] Running unit tests: pytest -m unit_tests'
        python -m pytest -m unit_tests

        echo '[INFO] Running converter validation: pytest ./tests/validation_tests/Converter_validation_test.py'
        python -m pytest ./tests/validation_tests/Converter_validation_test.py
      "; then
    echo "[OK] Base + dev + pytest succeeded in $img"
    ok+=("$img")
  else
    echo "[FAIL] Install/tests FAILED in $img"
    fail+=("$img")
  fi

  echo
done

echo "================ SUMMARY (Docker) ================"
if ((${#ok[@]} > 0)); then
  echo "Success (base + dev + pytest):"
  for img in "${ok[@]}"; do
    echo "  - $img"
  done
else
  echo "Success: none"
fi

echo
if ((${#fail[@]} > 0)); then
  echo "Failures:"
  for img in "${fail[@]}"; do
    echo "  - $img"
  done
else
  echo "Failures: none"
fi
echo "================================================="