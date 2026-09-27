"""Fix round 1 (task 4 review): the plan added the refusal of rejected keys in
ConnectivityExtractor but never cleaned the code that *produces* them. These pin
each producer so `opticonn tune-bayes` (and friends) do not hand
ConnectivityExtractor a config it is guaranteed to refuse.
"""

import dataclasses
from types import SimpleNamespace

import pytest

from scripts.extract_connectivity_matrices import (
    REJECTED_KEYS, tracking_config_errors)


def test_bayesian_param_space_has_no_rejected_parameters():
    from scripts.bayesian_optimizer import ParameterSpace
    names = {f.name for f in dataclasses.fields(ParameterSpace)}
    assert not names & {"track_voxel_ratio", "dt_threshold", "connectivity_threshold"}


def test_bayesian_iteration_config_is_executable_as_written(tmp_path):
    from scripts.bayesian_optimizer import BayesianOptimizer, ParameterSpace
    import json

    fake_self = SimpleNamespace(
        base_config={"tract_count": 50000, "thread_count": 1,
                     "connectivity_values": ["count"]},
        param_space=ParameterSpace(),
        iterations_dir=tmp_path,
    )
    config_path = BayesianOptimizer._create_config_for_params(fake_self, {}, 0)
    cfg = json.loads(config_path.read_text())
    assert tracking_config_errors(cfg) == []


def test_apply_unmapped_params_rejects_track_voxel_ratio():
    from scripts.cross_validation_bootstrap_optimizer import apply_unmapped_params
    with pytest.raises(ValueError, match="track_voxel_ratio"):
        apply_unmapped_params({}, {"track_voxel_ratio": 3.0}, {})


def test_apply_unmapped_params_rejects_dt_threshold():
    from scripts.cross_validation_bootstrap_optimizer import apply_unmapped_params
    with pytest.raises(ValueError, match="dt_threshold"):
        apply_unmapped_params({}, {"dt_threshold": 0.2}, {})


def test_apply_unmapped_params_rejects_connectivity_threshold():
    from scripts.cross_validation_bootstrap_optimizer import apply_unmapped_params
    with pytest.raises(ValueError, match="connectivity_threshold"):
        apply_unmapped_params({}, {"connectivity_threshold": 0.001}, {})


def test_apply_unmapped_params_still_applies_ordinary_keys():
    from scripts.cross_validation_bootstrap_optimizer import apply_unmapped_params
    out = apply_unmapped_params({}, {"fa_threshold": 0.2, "tract_count": 5000}, {})
    assert out["tracking_parameters"]["fa_threshold"] == 0.2
    assert out["tract_count"] == 5000


def test_merge_bayes_params_does_not_promote_rejected_keys(tmp_path):
    import json
    from scripts.cross_validation_bootstrap_optimizer import merge_bayes_params_into_config

    bayes_path = tmp_path / "bayes.json"
    bayes_path.write_text(json.dumps({"best_parameters": {
        "fa_threshold": 0.2, "track_voxel_ratio": 3.0, "connectivity_threshold": 0.001}}))
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"tract_count": 50000, "thread_count": 1,
                                     "connectivity_values": ["count"]}))

    seeded_path = merge_bayes_params_into_config(bayes_path, base_path, tmp_path)
    cfg = json.loads(seeded_path.read_text())
    assert tracking_config_errors(cfg) == []
