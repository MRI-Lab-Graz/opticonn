"""The grid must be able to express every parameter that measurably moves the
connectome, and must not enumerate cells that are the same specification twice.

A screen over three subjects (OpenNeuro ds003505, distance = 1-r on log1p
upper-triangle edge vectors, against a different-seed noise floor) found
tracking method to move the connectome 2.9x noise -- but `method` was not
sweepable at all. The same screen found `track_voxel_ratio` inert when
`tract_count` is fixed, which had already cost a sweep half its compute on
duplicate cells.
"""

from scripts.sweep_utils import (
    build_param_grid_from_config,
    drop_inert_cells,
    grid_product,
)


def _combos(sweep_parameters):
    """Mirrors what build_combos does: expand, enumerate, then drop inert cells.

    grid_product stays a generic cartesian product; the knowledge of which cells
    are the same specification lives in drop_inert_cells.
    """
    values, mapping = build_param_grid_from_config({"sweep_parameters": sweep_parameters})
    return drop_inert_cells(grid_product(values)), mapping


def test_tracking_method_is_sweepable():
    combos, mapping = _combos({"method_range": [0, 1]})
    assert sorted(c["method"] for c in combos) == [0, 1]
    assert mapping["method"] == "tracking_parameters.method"


def test_method_multiplies_the_grid_like_any_other_axis():
    combos, _ = _combos({"method_range": [0, 1], "turning_angle_range": [35, 65]})
    assert len(combos) == 4
    assert {(c["method"], c["turning_angle"]) for c in combos} == {
        (0, 35), (0, 65), (1, 35), (1, 65)}


def test_otsu_is_not_enumerated_when_fa_threshold_makes_it_inert():
    # DSI Studio uses otsu_threshold only when fa_threshold is 0; with fa > 0 the
    # otsu value is ignored, so fa=0.1/otsu=0.6 and fa=0.1/otsu=0.8 are one
    # specification run twice -- the track_voxel_ratio mistake in another guise.
    combos, _ = _combos({"fa_threshold_range": [0.0, 0.1], "otsu_range": [0.6, 0.8]})
    assert len(combos) == 3
    assert sorted((c["fa_threshold"], c["otsu_threshold"]) for c in combos) == [
        (0.0, 0.6), (0.0, 0.8), (0.1, 0.6)]


def test_collapsing_inert_otsu_keeps_the_other_axes_intact():
    combos, _ = _combos({"fa_threshold_range": [0.0, 0.1], "otsu_range": [0.6, 0.8],
                         "turning_angle_range": [35, 50, 65]})
    assert len(combos) == 9, "3 anisotropy levels x 3 angles"
    assert len({(c["fa_threshold"], c["otsu_threshold"], c["turning_angle"])
                for c in combos}) == 9


def test_otsu_alone_is_untouched():
    combos, _ = _combos({"otsu_range": [0.6, 0.8]})
    assert sorted(c["otsu_threshold"] for c in combos) == [0.6, 0.8]


def test_fa_alone_is_untouched():
    combos, _ = _combos({"fa_threshold_range": [0.0, 0.1]})
    assert sorted(c["fa_threshold"] for c in combos) == [0.0, 0.1]


def test_build_combos_drops_inert_cells_too():
    # The collapse must be in the path the sweep actually runs, not only in the
    # helper: build_combos is what cross_validation_bootstrap_optimizer calls.
    from scripts.cross_validation_bootstrap_optimizer import build_combos
    combos, sampler, _ = build_combos(
        {"fa_threshold_range": [0.0, 0.1], "otsu_range": [0.6, 0.8]}, None)
    assert sampler == "grid"
    assert len(combos) == 3, combos


def test_random_sampling_also_drops_inert_cells():
    # The grid path collapsed inert cells but random/LHS did not, so a config
    # using "sampling": {"method": "random"} still spent draws on cells that are
    # one specification. Verified empirically that they are one specification:
    # fa=0.1 with otsu 0.6 vs 0.8 gives bit-identical matrices (ds003505,
    # fixed seed, single thread, max abs diff 0.0).
    from scripts.cross_validation_bootstrap_optimizer import build_combos
    sp = {"fa_threshold_range": [0.1], "otsu_range": [0.6, 0.8],
          "sampling": {"method": "random", "n_samples": 8, "random_seed": 1}}
    combos, sampler, _ = build_combos(sp, None)
    assert sampler == "random"
    assert len(combos) == 1, f"fa=0.1 makes otsu inert, so there is one cell: {combos}"


def test_dropping_inert_cells_is_reported(caplog):
    # A silently smaller grid is the failure mode this repo keeps hitting. Several
    # shipped configs shrink when cells collapse (user_friendly_sweep 96 -> 48),
    # so the run must say so rather than let someone discover it by counting.
    import logging
    from scripts.cross_validation_bootstrap_optimizer import build_combos
    with caplog.at_level(logging.INFO):
        build_combos({"fa_threshold_range": [0.0, 0.1], "otsu_range": [0.6, 0.8]}, None)
    assert any("inert" in r.message.lower() for r in caplog.records), caplog.text


def test_a_grid_with_no_inert_cells_is_left_alone():
    from scripts.cross_validation_bootstrap_optimizer import build_combos
    combos, _, _ = build_combos({"turning_angle_range": [35, 50]}, None)
    assert len(combos) == 2
