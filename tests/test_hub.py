"""Tests for scripts/opticonn_hub.py CLI argument surface."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_analyze_help_drops_removed_flags():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.opticonn_hub", "analyze", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for flag in ("--skip-extraction", "--outlier-detection", "--interactive"):
        assert flag not in result.stdout
