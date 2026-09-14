import json

from scripts.cross_validation_bootstrap_optimizer import to_phase2_candidate


def _write_config(tmp_path, backend=None):
    cfg = {
        "tract_count": 500000,
        "tracking_parameters": {"fa_threshold": 0.1, "min_length": 10},
        "connectivity_options": {"connectivity_threshold": 0.001},
    }
    if backend is not None:
        cfg["backend"] = backend
    cfg_path = tmp_path / "sweep_0001.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def _row(cfg_path):
    return {
        "config_path": str(cfg_path),
        "atlas": "FreeSurferDKT_Cortical",
        "connectivity_metric": "count",
        "discriminability": 0.83,
        "repeatability": 0.91,
        "loo_top1_frequency": 0.75,
    }


def test_to_phase2_candidate_shapes_row(tmp_path):
    cfg_path = _write_config(tmp_path, backend="dsi_studio")
    row = _row(cfg_path)

    result = to_phase2_candidate(row)

    assert result == {
        "atlas": "FreeSurferDKT_Cortical",
        "connectivity_metric": "count",
        "average_score": 0.83,
        "repeatability": 0.91,
        "loo_top1_frequency": 0.75,
        "backend": "dsi_studio",
        "parameters": {
            "tract_count": 500000,
            "tracking_parameters": {"fa_threshold": 0.1, "min_length": 10},
            "connectivity_threshold": 0.001,
        },
    }


def test_to_phase2_candidate_defaults_backend(tmp_path):
    cfg_path = _write_config(tmp_path, backend=None)
    row = _row(cfg_path)

    result = to_phase2_candidate(row)

    assert result["backend"] == "dsi_studio"
