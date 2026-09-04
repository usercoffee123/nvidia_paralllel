"""Integration tests for the command-line entrypoint."""
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = PROJECT_ROOT / "python" / "worker.py"


def _run(args):
    return subprocess.run(
        [sys.executable, str(WORKER_PATH), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def test_main_returns_error_for_bad_rows(worker_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["worker.py", "--rows", "0", "--cols", "3"])
    assert worker_module.main() == 1


def test_main_returns_error_for_bad_cols(worker_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["worker.py", "--rows", "10", "--cols", "0"])
    assert worker_module.main() == 1


def test_main_returns_error_for_bad_layers(worker_module, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["worker.py", "--rows", "10", "--cols", "3", "--qrc-layers", "0"]
    )
    assert worker_module.main() == 1


@pytest.mark.slow
def test_cli_smoke_run():
    result = _run(["--rows", "4", "--cols", "3"])
    assert result.returncode == 0, result.stderr
    assert "Dataset: 4x3" in result.stdout
    assert "CUDA-Q runtime:" in result.stdout


@pytest.mark.slow
def test_cli_output_lines_match_jobs():
    result = _run(["--rows", "3", "--cols", "2"])
    assert result.returncode == 0, result.stderr
    runtime = [ln for ln in result.stdout.splitlines() if ln.startswith("CUDA-Q runtime:")]
    assert len(runtime) == 1
