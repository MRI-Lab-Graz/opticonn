import json
import subprocess
from pathlib import Path

import pytest

from scripts import extract_connectivity_matrices as ecm
from scripts.extract_connectivity_matrices import (
    ConnectivityExtractor, build_track_command, tracking_config_errors)


def _cfg(**tracking):
    return {"tract_count": 50000, "thread_count": 1, "connectivity_values": ["count", "qa"],
            "tracking_parameters": tracking}


def _flags(cmd):
    return {t.split("=", 1)[0] for t in cmd if t.startswith("--")}


def test_null_is_omitted_and_every_literal_is_sent_even_zero():
    cmd = build_track_command(_cfg(turning_angle=0, step_size=None, fa_threshold=0.0,
                                   otsu_threshold=0.6, random_seed=1),
                              "dsi", "s.fz", "o.tt.gz", "AAL3")
    assert "--turning_angle=0" in cmd
    assert "--fa_threshold=0.0" in cmd
    assert "--otsu_threshold=0.6" in cmd       # formerly dropped as an 'assumed default'
    assert "--step_size" not in _flags(cmd)


def test_all_null_sends_no_tracking_flag():
    cmd = build_track_command(_cfg(**dict.fromkeys(ecm.DEFAULT_CONFIG["tracking_parameters"])),
                              "dsi", "s.fz", "o.tt.gz", "AAL3")
    tracking = set(ecm.DEFAULT_CONFIG["tracking_parameters"])
    assert not {f[2:] for f in _flags(cmd)} & tracking


def test_connectivity_threshold_is_never_sent():
    cfg = {**_cfg(), "connectivity_options": {"connectivity_type": "pass"}}
    assert "--connectivity_threshold" not in _flags(
        build_track_command(cfg, "dsi", "s.fz", "o.tt.gz", "AAL3"))


def test_defaults_carry_no_assumed_values():
    assert all(v is None for v in ecm.DEFAULT_CONFIG["tracking_parameters"].values())
    assert "connectivity_threshold" not in ecm.DEFAULT_CONFIG["connectivity_options"]


@pytest.mark.parametrize("key", ["track_voxel_ratio", "dt_threshold"])
def test_parameters_dsi_studio_would_not_apply_are_errors(key):
    errors = tracking_config_errors(_cfg(**{key: 1.0}))
    assert len(errors) == 1 and key in errors[0]


def test_unknown_tracking_key_is_an_error():
    assert "unknown" in tracking_config_errors(_cfg(fa_treshold=0.1))[0]


def test_connectivity_threshold_is_an_error():
    cfg = {**_cfg(), "connectivity_options": {"connectivity_threshold": 0.001}}
    assert "connectivity_threshold" in tracking_config_errors(cfg)[0]


def test_extractor_refuses_a_config_it_cannot_execute_faithfully():
    with pytest.raises(ValueError, match="dt_threshold"):
        ConnectivityExtractor(_cfg(dt_threshold=0.2))


def test_null_tracking_values_do_not_crash_validation_or_naming(tmp_path):
    ex = ConnectivityExtractor(_cfg(**dict.fromkeys(ecm.DEFAULT_CONFIG["tracking_parameters"])))
    run_dir = ex.create_output_structure(str(tmp_path), "sub01")
    assert run_dir.name == "tracks_50k_streamline"


def test_json_validator_accepts_null_tracking_values():
    # The reference combo's derived config reaches this validator via
    # run_pipeline.load_test_configuration; before the fix, None < 0.0 raised TypeError.
    from scripts.json_validator import JSONValidator
    cfg = {"atlases": ["AAL3"], "connectivity_values": ["count"], "tract_count": 50000,
           "thread_count": 1,
           "tracking_parameters": {"fa_threshold": None, "turning_angle": None,
                                   "otsu_threshold": None, "min_length": None,
                                   "max_length": None}}
    errors = JSONValidator()._validate_dsi_studio_config(cfg, dry_run=True)
    assert not [e for e in errors if any(k in e for k in (
        "fa_threshold", "turning_angle", "otsu_threshold", "min_length", "max_length"))], errors


@pytest.mark.parametrize("axis", ["connectivity_threshold_range", "track_voxel_ratio_range",
                                  "dt_threshold_range"])
def test_sweeping_an_unapplied_parameter_is_refused(axis):
    from scripts.sweep_utils import build_param_grid_from_config
    with pytest.raises(ValueError, match=axis):
        build_param_grid_from_config({"sweep_parameters": {axis: [1, 2]}})


# The schema file is not valid JSON (pre-existing; out of scope) and is not a config.
# Shipped configs only: tracked files. configs/ may also hold untracked private
# configs (.gitignore ignores configs/study*.json), which are not shipped.
_REPO = Path(__file__).parent.parent
_TRACKED = subprocess.run(["git", "ls-files", "configs/*.json"], cwd=_REPO,
                          capture_output=True, text=True, check=True).stdout.split()
CONFIGS = sorted(_REPO / p for p in _TRACKED if Path(p).name != "dsi_studio_config_schema.json")


def test_shipped_config_list_is_not_empty():
    assert len(CONFIGS) >= 10


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_is_executable_as_written(path):
    from scripts.sweep_utils import build_param_grid_from_config
    cfg = json.loads(path.read_text())
    assert tracking_config_errors(cfg) == []
    build_param_grid_from_config(cfg)   # raises on a sweep axis DSI Studio would not apply
