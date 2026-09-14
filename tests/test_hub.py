"""Tests for scripts/opticonn_hub.py CLI argument surface."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.cross_validation_bootstrap_optimizer import to_phase2_candidate
from scripts.opticonn_hub import candidate_to_extraction_config

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


def test_candidate_to_extraction_config_round_trips_full_settings(tmp_path):
    cfg = {
        "tract_count": 500000,
        "tracking_parameters": {
            "otsu_threshold": 0.5,
            "method": 1,
            "check_ending": 1,
            "fa_threshold": 0.1,
        },
        "connectivity_options": {
            "connectivity_threshold": 0.001,
            "connectivity_type": "pass",
        },
    }
    cfg_path = tmp_path / "sweep_0001.json"
    cfg_path.write_text(json.dumps(cfg))
    row = {
        "config_path": str(cfg_path),
        "atlas": "FreeSurferDKT_Cortical",
        "connectivity_metric": "count",
        "discriminability": 0.9,
        "repeatability": 0.9,
        "loo_top1_frequency": 0.5,
    }
    candidate = to_phase2_candidate(row)

    extraction_cfg = candidate_to_extraction_config(candidate, "dsi_studio")

    assert extraction_cfg["atlases"] == ["FreeSurferDKT_Cortical"]
    assert extraction_cfg["connectivity_values"] == ["count"]
    assert extraction_cfg["dsi_studio_cmd"] == "dsi_studio"
    assert extraction_cfg["backend"] == "dsi_studio"
    assert extraction_cfg["tract_count"] == 500000
    assert extraction_cfg["tracking_parameters"] == {
        "otsu_threshold": 0.5,
        "method": 1,
        "check_ending": 1,
        "fa_threshold": 0.1,
    }
    assert extraction_cfg["connectivity_options"] == {
        "connectivity_threshold": 0.001,
        "connectivity_type": "pass",
    }
