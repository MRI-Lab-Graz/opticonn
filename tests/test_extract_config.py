from scripts.extract_connectivity_matrices import resolve_config


def test_sweep_tract_count_reaches_dsi_track_count():
    assert resolve_config({"tract_count": 2500000})["track_count"] == 2500000


def test_default_track_count_without_tract_count():
    assert resolve_config({})["track_count"] == 100000


def test_nested_tracking_parameters_merge_over_defaults():
    cfg = resolve_config({"tracking_parameters": {"fa_threshold": 0.2}})
    assert cfg["tracking_parameters"]["fa_threshold"] == 0.2
    assert cfg["tracking_parameters"]["otsu_threshold"] == 0.6
