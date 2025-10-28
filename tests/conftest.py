# conftest.py
from __future__ import annotations

import os, shutil, uuid, pytest
from pathlib import Path
from typing import Iterator


def _compute_base_tmp(worker_id: str) -> Path:
    """Compute a per-job, per-shard, per-worker temp root.

    Layout:
        <BASE>/<shard_tag>/<worker_id>

    Where:
        BASE: $SHINIER_BASE_TMP or <repo>/.pytest_tmp
        shard_tag: "shard<N>-of-<M>" if SHARDS/SHARD_INDEX set, else "shard0-of-1"
        worker_id: "master" (no xdist) or "gw<N>" (xdist)

    Args:
        worker_id: Worker identifier (e.g., "master" or "gw0").

    Returns:
        Path: The worker-scoped temp root.
    """
    # Base root (override via env if you like putting temps on a faster volume)
    default_base = Path(__file__).resolve().parent / "IMAGES" / "tmp"
    base = Path(os.getenv("SHINIER_BASE_TMP", default_base)).resolve()

    # Shard namespace (job-level parallelism)
    shards = int(os.getenv("SHARDS", "1"))
    shard_index = int(os.getenv("SHARD_INDEX", "0"))
    shard_tag = f"shard{shard_index}-of-{shards}"

    return base / shard_tag / worker_id


def _safe_rmtree(path: Path) -> None:
    """Remove a path if it exists, handling common filesystem errors safely.

    Only handles expected errors; unexpected exceptions are allowed to surface.

    Args:
        path: Directory to remove recursively.
    """
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except PermissionError:
        # Retry with onerror to chmod problematic nodes then remove.
        def _onerror(func, p, exc_info):  # type: ignore[no-untyped-def]
            try:
                os.chmod(p, 0o700)
                func(p)
            except FileNotFoundError:
                pass
            except PermissionError:
                # Give up quietly; usually a stale tmp artifact on CI.
                pass

        shutil.rmtree(path, ignore_errors=True, onerror=_onerror)  # type: ignore[arg-type]


@pytest.fixture(scope="session", autouse=True)
def _session_tmp_root(request) -> Path:
    """Create/clean the per-worker temp root at session start.

    Only cleans THIS worker's root, so running multiple shards/jobs in parallel
    remains safe. Each shard/job writes to its own shard-tagged directory.

    Args:
        request: Pytest request object (provides xdist worker info if present).

    Returns:
        Path: The per-worker temp root for this session.
    """
    # xdist exposes "workerid" (note: no underscore). Fall back to "master".
    worker_input = getattr(request.config, "workerinput", {}) or {}
    worker_id = worker_input.get("workerid", "master")

    root = _compute_base_tmp(worker_id)

    # Nuke and recreate JUST this worker's root.
    _safe_rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    return root


@pytest.fixture
def test_tmpdir(_session_tmp_root: Path) -> Iterator[Path]:
    """Function-scoped unique temp dir under the per-worker session root.

    Example:
        def test_something(test_tmpdir):
            inp = test_tmpdir / "INPUT"
            inp.mkdir()
            # ... test code ...

    Args:
        _session_tmp_root: The per-worker session root (from session fixture).

    Yields:
        Path: A unique directory for this test function.
    """
    d: Path = _session_tmp_root / f"case-{uuid.uuid4().hex[:12]}"
    d.mkdir(parents=True, exist_ok=False)
    try:
        yield d
    finally:
        # Clean up after the test; comment this out if you want to inspect.
        _safe_rmtree(d)
