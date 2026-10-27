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


from scripts.report_stats import effective_dimensionality, fragility, reference_displacement, reference_rho


def _scans(values, scale=1.0):
    return {f"sub-{i + 1:02d}_ses-1_dwi": {"m": v * scale} for i, v in enumerate(values)}


def test_fragility_contrasts_repeats_of_one_specification_with_different_specifications():
    up = [1.0, 2.0, 3.0, 4.0, 5.0]
    per = {"w1/s1": {k: [m, dict(m)] for k, m in _scans(up).items()},                 # repeats identical
           "w1/s2": {k: [m, dict(m)] for k, m in _scans(up[::-1]).items()},          # reversed ordering
           "w2/s1": {k: [m, dict(m)] for k, m in _scans(up).items()}}                # alone in its wave
    got = fragility(per)
    assert got["seed_rho"] == pytest.approx(1.0) and got["param_rho"] == pytest.approx(-1.0)
    assert got["gap"] == pytest.approx(2.0)
    assert got["n_param_pairs"] == 1                   # w1/s1 with w1/s2 only: no pairing across waves
    assert got["n_seed_pairs"] == 3 and got["n_measures"] == 1 and got["min_n_subjects"] == 5


def test_fragility_skips_specifications_with_too_few_subjects():
    per = {"w1/s1": {k: [m, dict(m)] for k, m in _scans([1.0, 2.0]).items()}}
    got = fragility(per)
    assert np.isnan(got["seed_rho"]) and got["n_seed_pairs"] == 0


def _two_repeat(values_by_measure, n):
    rng = np.random.default_rng(0)
    per = {}
    for c in range(n // 15):
        per[f"w1/s{c}"] = {f"sub-{i:02d}_ses-1_dwi": [{k: float(v[c * 15 + i]) for k, v in values_by_measure.items()}] * 2
                           for i in range(15)}
    return per


def test_effective_dimensionality_is_one_for_perfectly_correlated_measures_and_the_count_for_independent_ones():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(600)
    corr = {"a": x, "b": 2 * x + 1, "c": -x}
    indep = {"a": rng.standard_normal(600), "b": rng.standard_normal(600), "c": rng.standard_normal(600)}
    assert effective_dimensionality(_two_repeat(corr, 600))["effective_dimensionality"] == pytest.approx(1.0, abs=1e-6)
    got = effective_dimensionality(_two_repeat(indep, 600))
    assert got["effective_dimensionality"] == pytest.approx(3.0, rel=0.1)
    assert got["n_ed_rows"] == 600 and got["n_ed_subjects"] == 15 and got["n_ed_dropped"] == 0


def test_effective_dimensionality_drops_non_finite_rows_and_needs_enough_subjects():
    per = _two_repeat({"a": np.arange(30.0), "b": np.arange(30.0) ** 2}, 30)
    first = next(iter(per["w1/s0"]))
    per["w1/s0"][first] = [{"a": float("nan"), "b": 1.0}] * 2
    assert effective_dimensionality(per)["n_ed_dropped"] == 1
    tiny = {"w1/s0": {f"sub-0{i}_ses-1_dwi": [{"a": float(i), "b": 1.0}] * 2 for i in range(1, 3)}}
    assert np.isnan(effective_dimensionality(tiny)["effective_dimensionality"])


def test_reference_rho_is_the_agreement_of_each_candidate_with_the_reference_in_its_wave():
    ref = _scans([1.0, 2.0, 3.0, 4.0])
    rep1 = {"w1/ref": ref, "w1/s1": _scans([1.0, 2.0, 3.0, 4.0], 2.0), "w1/s2": _scans([4.0, 3.0, 2.0, 1.0]),
            "w2/s1": _scans([1.0, 2.0, 3.0, 4.0])}                       # its wave has no reference
    got = reference_rho(rep1, {"w1/ref"})
    assert got["reference_rho"] == pytest.approx(0.0) and got["n_reference_ordering_pairs"] == 2
    assert reference_rho({"w1/s1": ref}, set())["reference_rho"] != reference_rho({"w1/s1": ref}, set())["reference_rho"]  # NaN
    assert "no reference" in reference_rho({"w1/s1": ref}, set())["note"]


def test_displacement_is_the_distance_to_the_reference_relative_to_the_between_subject_distance():
    s1, s2 = np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0])
    c1, c2 = np.array([1.0, 2.0, 4.0, 3.0]), np.array([4.0, 3.0, 1.0, 2.0])
    k1, k2 = "sub-01_ses-1_dwi", "sub-02_ses-1_dwi"
    vectors = {"w1/ref": {k1: s1, k2: s2}, "w1/cand": {k1: c1, k2: c2}}
    got = reference_displacement(vectors, {"w1/ref"})
    # candidate vs reference: 1 - 0.8 = 0.2 for both subjects; between subjects: 1 - (-1) = 2 in both specifications
    assert got["available"] is True and got["n_reference_pairs"] == 2
    assert got["reference_over_between_subject"] == pytest.approx(0.1)


def test_displacement_without_a_reference_is_reported_as_unavailable():
    got = reference_displacement({"w1/s1": {"sub-01_ses-1_dwi": np.array([1.0, 2.0, 3.0])}}, set())
    assert got["available"] is False and got["reference_over_between_subject"] is None and got["note"]


def test_how_close_the_top_is_is_the_rank_of_each_winner_in_the_other_wave_and_the_top_five_overlap():
    m1 = {s: 1.0 - 0.1 * i for i, s in enumerate("abcdefg")}                    # a > b > ... > g
    m2 = dict(m1, a=0.75, c=0.95)                                               # wave 2: c > b > a > d > ...
    got = leader_test(m1, m2, [1.0])
    assert got["winner_wave1"] == "a" and got["winner_wave2"] == "c"
    assert got["winner_wave1_rank_in_wave2"] == 3 and got["winner_wave2_rank_in_wave1"] == 3
    assert got["top5_overlap"] == 5                                              # {a,b,c,d,e} in both


def test_the_rank_fields_are_undefined_below_three_shared_candidates():
    got = leader_test({"a": 0.3, "b": 0.2}, {"a": 0.3, "b": 0.2}, [1.0])
    assert got["winner_wave1_rank_in_wave2"] is None and got["top5_overlap"] is None
