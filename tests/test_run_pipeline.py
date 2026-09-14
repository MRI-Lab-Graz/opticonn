"""Tests for scripts/run_pipeline.py CLI behavior (Phase 2: extraction + aggregation only)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_pipeline(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_pipeline", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_step_02_rejected_by_argparse(tmp_path):
    result = run_pipeline(["--step", "02", "--output", str(tmp_path)])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_dry_run_step_all_shows_extraction_and_aggregation_only(tmp_path):
    result = run_pipeline(
        [
            "--dry-run",
            "--step",
            "all",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--extraction-config",
            "configs/quick_sweep.json",
        ]
    )
    assert result.returncode == 0
    assert "[DRY-RUN] Would run:" in result.stdout
    assert "scripts.extract_connectivity_matrices" in result.stdout
    assert "scripts.aggregate_network_measures" in result.stdout
    assert "metric_optimizer" not in result.stdout
    assert "optimal_selection" not in result.stdout


def test_dry_run_step_01_has_no_aggregate_command(tmp_path):
    result = run_pipeline(
        [
            "--dry-run",
            "--step",
            "01",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--extraction-config",
            "configs/quick_sweep.json",
        ]
    )
    assert result.returncode == 0
    assert "scripts.extract_connectivity_matrices" in result.stdout
    assert "scripts.aggregate_network_measures" not in result.stdout
