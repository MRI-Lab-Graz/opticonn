import pytest

from scripts import dsi_preflight as pf

SWEEP = {
    "tract_count": 50000, "thread_count": 1, "connectivity_values": ["count"],
    "reliability": {"repeats": 2},
    "tracking_parameters": {"step_size": 1.0, "smoothing": 0.1, "max_length": 250},
    "sweep_parameters": {
        "fa_threshold_range": [0.0, 0.1], "otsu_range": [0.6, 0.8],
        "turning_angle_range": [35, 50],
        "reference_candidate": {"fa_threshold": None, "turning_angle": None,
                                "step_size": None, "smoothing": None, "max_length": None},
    },
}


def test_enumerates_every_distinct_spec_and_repeat():
    specs = pf.enumerate_specs(SWEEP)
    # (fa 0/otsu 0.6, fa 0/otsu 0.8, fa 0.1) x 2 angles = 6 grid + 1 reference, x 2 repeats
    assert len(specs) == 14
    assert {s["repeat"] for s in specs} == {1, 2}
    assert all(s["config"]["tracking_parameters"]["random_seed"] == s["repeat"] for s in specs)


def test_reference_spec_carries_nulls_through_to_its_config():
    ref = [s for s in pf.enumerate_specs(SWEEP) if s["choice"].get("turning_angle") is None]
    assert ref and all(s["config"]["tracking_parameters"]["turning_angle"] is None for s in ref)


def test_preflight_requires_single_threaded_determinism():
    with pytest.raises(ValueError, match="thread_count"):
        pf.enumerate_specs({**SWEEP, "thread_count": 4})


def test_single_axis_pairs():
    choices = [{"a": 1, "b": 1}, {"a": 2, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 2}]
    assert sorted(pf.single_axis_pairs(choices)) == [
        (0, 1, "a"), (0, 2, "b"), (1, 3, "b"), (2, 3, "a")]


def _run(spec, repeat, key, pid="PID", errors=()):
    return {"spec": spec, "repeat": repeat, "fingerprint_key": key, "parameter_id": pid,
            "errors": list(errors)}


def test_summary_passes_only_when_everything_passed():
    runs = [_run(0, 1, "k0", "p0"), _run(1, 1, "k1", "p1")]
    s = pf.summarize(runs, [(0, 1, "a")], {0: "d0", 1: "d1"})
    assert s["passed"] is True
    assert s["expected_fingerprints"] == {"k0": "p0", "k1": "p1"}


def test_any_run_error_fails_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", errors=["boom"])], [], {0: "d0"})
    assert s["passed"] is False and "boom" in s["failures"][0]


def test_an_axis_that_changes_nothing_fails_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", "p0"), _run(1, 1, "k1", "p1")],
                     [(0, 1, "turning_angle")], {0: "same", 1: "same"})
    assert s["passed"] is False
    assert "turning_angle" in s["failures"][0] and "did not change" in s["failures"][0]


def test_two_specs_with_one_fingerprint_fail_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", "p"), _run(1, 1, "k1", "p")], [], {0: "a", 1: "b"})
    assert s["passed"] is False and "same parameter_id" in s["failures"][0]
