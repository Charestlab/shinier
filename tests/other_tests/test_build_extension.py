# tests/validation_tests/test_build_extension.py
import subprocess
import sys
from pathlib import Path


def test_project_builds():
    """Ensure the package (including C++ extension) builds successfully."""
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    # If compilation fails, returncode != 0 and you get the compiler errors in the assertion.
    assert result.returncode == 0, (
        "Building the project (C++ extension) failed:\n"
        + result.stdout
    )