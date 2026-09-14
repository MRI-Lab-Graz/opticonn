import json

from scripts.cross_validation_bootstrap_optimizer import (
    filter_requested_metrics,
    json_safe,
    to_phase2_candidate,
)


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
            "connectivity_options": {"connectivity_threshold": 0.001},
        },
    }


def test_to_phase2_candidate_defaults_backend(tmp_path):
    cfg_path = _write_config(tmp_path, backend=None)
    row = _row(cfg_path)

    result = to_phase2_candidate(row)

    assert result["backend"] == "dsi_studio"


def test_filter_requested_metrics_keeps_only_requested():
    rows = [
        {"connectivity_metric": "count"},
        {"connectivity_metric": "fa"},
        {"connectivity_metric": "qa"},
    ]
    kept = filter_requested_metrics(rows, {"connectivity_values": ["count", "qa"]})
    assert [r["connectivity_metric"] for r in kept] == ["count", "qa"]


def test_filter_requested_metrics_keeps_all_when_unset():
    rows = [{"connectivity_metric": "count"}, {"connectivity_metric": "fa"}]
    assert filter_requested_metrics(rows, {}) == rows
    assert filter_requested_metrics(rows, {"connectivity_values": []}) == rows


def test_json_safe_replaces_non_finite_floats_recursively():
    data = {
        "a": float("nan"),
        "b": [1.0, float("inf"), float("-inf")],
        "c": {"d": 2.5, "e": float("nan")},
        "f": "text",
        "g": 3,
    }
    result = json_safe(data)
    assert result == {
        "a": None,
        "b": [1.0, None, None],
        "c": {"d": 2.5, "e": None},
        "f": "text",
        "g": 3,
    }
    json.dumps(result)  # must not raise and must not emit bare NaN
