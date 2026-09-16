"""Regression test for `opticonn apply` validating the wrong file.

`opticonn apply --optimal-config <bayesian_optimization_results.json>` is the
documented Step 3 of the recommended workflow (see opticonn.py --help and
docs/demos.md). That results file only ever contains best_parameters/scores
- it never has 'atlases' or 'connectivity_values' by design, since select
just points back at it as-is.

scripts/opticonn_hub.py's apply handler correctly builds a *merged*
extraction config (default config + best_parameters) at
`out_selected / "final_extraction_config.json"` and uses that to run the
real pipeline - but then validates `args.optimal_config` (the original,
deliberately atlas-less results file) instead of the merged file it just
built, so every apply invocation following the documented workflow fails
config validation before doing anything.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_apply_does_not_validate_the_raw_bayesian_results_file(tmp_path):
    optimal_config = tmp_path / "bayesian_optimization_results.json"
    optimal_config.write_text(
        json.dumps(
            {
                "target_modality": "fa",
                "best_parameters": {
                    "tract_count": 10000,
                    "fa_threshold": 0.1,
                    "turning_angle": 60,
                    "step_size": 1.0,
                },
                "best_quality_score": 0.5,
            }
        )
    )

    empty_data_dir = tmp_path / "no_such_data"
    empty_data_dir.mkdir()

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "opticonn.py"),
            "apply",
            "--data-dir",
            str(empty_data_dir),
            "--optimal-config",
            str(optimal_config),
            "-o",
            str(tmp_path / "out"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined = proc.stdout + proc.stderr
    # The bug validated the raw, deliberately atlas-less results file instead
    # of the merged extraction config apply itself builds; assert those
    # specific missing-field errors are gone. (A DSI-Studio-not-found error
    # here is an unrelated, expected environment limitation in CI/test
    # sandboxes without DSI Studio installed - not what this test guards.)
    assert "'atlases' field is required" not in combined, combined
    assert "'connectivity_values' field is required" not in combined, combined
