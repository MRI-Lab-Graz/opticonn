"""CLI behavior tests: DSI Studio precedence, dry-run placement, venv check."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dsi_studio_flag_wins_over_config():
    """--dsi-studio must override any DSI_STUDIO_CMD set via .opticonn_config."""
    env = {**os.environ, "DSI_STUDIO_CMD": ""}
    result = subprocess.run(
        [
            sys.executable,
            "opticonn.py",
            "--dsi-studio",
            "/nope",
            "sweep",
            "--data",
            "examples/data/fib_samples",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "DSI Studio not found at: /nope" in combined


def test_dsi_studio_flag_wins_when_config_path_is_missing():
    """--dsi-studio must apply even when --config points at a nonexistent file.

    Old code only ever set DSI_STUDIO_CMD from the flag inside a branch
    gated on `cfg_path_candidate.exists()`, so a missing --config path
    silently dropped the flag and fell through to .opticonn_config.
    """
    env = {k: v for k, v in os.environ.items() if k != "DSI_STUDIO_CMD"}
    result = subprocess.run(
        [
            sys.executable,
            "opticonn.py",
            "--dsi-studio",
            "/nope",
            "sweep",
            "--config",
            "/nonexistent/sweep.json",
            "--data",
            "examples/data/fib_samples",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    combined = result.stdout + result.stderr
    assert "DSI Studio not found at: /nope" in combined


def test_validate_environment_no_issue_inside_virtualenv(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", "/fake/venv")
    monkeypatch.setattr(sys, "base_prefix", "/fake/base")
    import opticonn

    importlib.reload(opticonn)
    _, issues = opticonn.validate_environment()
    assert not any("virtual environment" in i.lower() for i in issues)


def test_validate_environment_reports_issue_outside_virtualenv(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "prefix", "/fake/same")
    monkeypatch.setattr(sys, "base_prefix", "/fake/same")
    import opticonn

    importlib.reload(opticonn)
    _, issues = opticonn.validate_environment()
    assert any("virtual environment" in i.lower() for i in issues)


def test_dry_run_after_subcommand():
    result = subprocess.run(
        [
            sys.executable,
            "opticonn.py",
            "sweep",
            "--data",
            "examples/data/fib_samples",
            "--quick",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert "[DRY-RUN] Would execute" in combined


def test_global_dry_run_removed():
    result = subprocess.run(
        [sys.executable, "opticonn.py", "--dry-run", "validate"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_validate_backend_mrtrix3_checks_mrtrix_tools_only(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "DSI_STUDIO_CMD"}
    env["PATH"] = str(tmp_path)  # no tckgen / tck2connectome here
    out = subprocess.run(
        [sys.executable, "opticonn.py", "validate", "--backend", "mrtrix3"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    text = out.stdout + out.stderr
    assert out.returncode == 1
    assert "MRtrix3 `tckgen` not found on PATH" in text
    assert "DSI Studio" not in text and "DSI_STUDIO_CMD" not in text


def test_validate_rejects_unknown_backend():
    out = subprocess.run(
        [sys.executable, "opticonn.py", "validate", "--backend", "fsl"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "invalid choice" in out.stderr
