# tests/test_report_stats.py
import numpy as np
import pytest

from scripts.report_stats import density_relation, leader_test, reference_percentile, wave_of


def test_wave_of_splits_a_scoped_id_and_leaves_a_plain_id_alone():
    assert wave_of("wave1/sweep_0003") == "wave1" and wave_of("sweep_0003") == ""


def test_a_determined_leader_has_a_gap_larger_than_the_wave_to_wave_variation():
    m1 = {"a": 0.90, "b": 0.50, "c": 0.30, "d": 0.10}
    m2 = {"a": 0.88, "b": 0.52, "c": 0.31, "d": 0.09}
    got = leader_test(m1, m2, [1.0, 1.0])
    assert got["n_candidates"] == 4 and got["discriminability_saturated"] is True
    assert got["cross_wave_rho"] == pytest.approx(1.0)
    assert got["winner_wave1"] == got["winner_wave2"] == "a" and got["winners_agree"] is True
    assert got["top_two_separation"] == pytest.approx(0.38) and got["wave_variation"] == pytest.approx(0.015)
    assert got["winner_determined"] is True and got["notes"] == []


def test_an_undetermined_leader_and_disagreeing_winners():
    m1 = {"a": 0.300, "b": 0.299, "c": 0.100}
    m2 = {"a": 0.250, "b": 0.251, "c": 0.050}
    got = leader_test(m1, m2, [1.0, 0.9])
    assert got["winners_agree"] is False and got["winner_determined"] is False
    assert got["discriminability_saturated"] is False
    assert got["top_two_separation"] < got["wave_variation"]


def test_a_candidate_with_a_missing_margin_is_excluded_and_few_candidates_skip_the_rest():
    got = leader_test({"a": 0.3, "b": 0.2, "c": None}, {"a": 0.3, "b": float("nan"), "c": 0.1}, [])
    assert got["n_candidates"] == 1 and got["winner_wave1"] is None and got["winner_determined"] is None
    assert got["discriminability_saturated"] is None
    assert any("fewer than 3" in n for n in got["notes"]) and any("discriminability" in n for n in got["notes"])


def test_a_constant_ordering_has_no_cross_wave_rho_but_still_a_winner():
    got = leader_test({"a": 0.2, "b": 0.2, "c": 0.2}, {"a": 0.1, "b": 0.2, "c": 0.3}, [1.0])
    assert got["cross_wave_rho"] is None and any("constant" in n for n in got["notes"])
    assert got["winner_wave1"] == "a" and got["winner_wave2"] == "c"        # ties go to the first


def test_density_relation_is_the_spearman_of_each_measure_against_density_over_kept_specifications():
    means = {f"s{i}": {"density": 0.1 * i, "sw": 5 - i, "flat": 1.0} for i in range(1, 6)}
    means["ref"] = {"density": 0.9, "sw": 100.0, "flat": 1.0}
    got = density_relation(means, {"ref"})
    assert got["sw_vs_density"] == pytest.approx(-1.0)
    assert got["flat_vs_density"] is None and "density_vs_density" not in got


def test_reference_percentile_is_the_mean_over_waves_of_the_share_of_less_dense_specifications():
    by_wave = {"w1": {"a": 0.1, "b": 0.2, "c": 0.3, "ref": 0.25}, "w2": {"a": 0.1, "b": 0.2, "ref": 0.5}}
    got = reference_percentile(by_wave, {"w1": "ref", "w2": "ref"})
    assert got == pytest.approx((2 / 3 + 1.0) / 2)
    assert reference_percentile(by_wave, {}) is None
