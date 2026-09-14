from scripts.sweep_utils import apply_param_choice_to_config, build_param_grid_from_config, grid_product

BASE = {
    "tract_count": 100000,
    "tracking_parameters": {"fa_threshold": 0.1, "min_length": 10},
    "sweep_parameters": {"fa_threshold_range": [0.05, 0.2], "tract_count_range": [100000, 500000]},
}


def test_choice_lands_in_place_without_touching_base():
    out = apply_param_choice_to_config(BASE, {"fa_threshold": 0.2, "tract_count": 500000}, {})
    assert out["tracking_parameters"]["fa_threshold"] == 0.2
    assert out["tract_count"] == 500000
    assert BASE["tracking_parameters"]["fa_threshold"] == 0.1
    assert "sweep_parameters" not in out


def test_grid_covers_all_combinations():
    values, _ = build_param_grid_from_config(BASE)
    assert len(grid_product(values)) == 4
