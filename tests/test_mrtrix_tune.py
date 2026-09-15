"""Tests for the repeat-run discriminability rewiring in scripts/mrtrix_tune.py.

Mirrors tests/test_cross_validation_repeats.py's pattern: test the pure
selection function and the repeats-aware scoring function against fake data,
not against real tckgen/tck2connectome subprocess execution.
"""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.mrtrix_tune import Bundle, _compute_qa_for_theta, evaluate_theta, select_best_theta


def _theta(theta_id, discriminability, repeatability, quality_score_raw, rejected=""):
    return {
        "theta_id": theta_id,
        "discriminability": discriminability,
        "repeatability": repeatability,
        "quality_score_raw": quality_score_raw,
        "rejected": rejected,
    }


# ---------------------------------------------------------------------------
# select_best_theta
# ---------------------------------------------------------------------------


def test_selects_highest_discriminability_even_when_quality_score_raw_is_lower():
    thetas = [
        _theta("low_discr_high_quality", 0.55, 0.9, 0.99),
        _theta("high_discr_low_quality", 0.92, 0.8, 0.10),
    ]

    winner = select_best_theta(thetas)

    assert winner["theta_id"] == "high_discr_low_quality"


def test_ties_on_discriminability_break_by_repeatability():
    thetas = [
        _theta("tied_low_repeatability", 0.8, 0.5, 0.5),
        _theta("tied_high_repeatability", 0.8, 0.7, 0.5),
    ]

    winner = select_best_theta(thetas)

    assert winner["theta_id"] == "tied_high_repeatability"


def test_falls_back_to_repeatability_when_every_theta_has_nan_discriminability():
    # mrtrix_tune's CLI evaluates one --subject per run, so
    # scripts.reliability.discriminability (which needs >=2 subjects) is
    # structurally NaN for every theta -- rank() alone would exclude every
    # row and return nothing.  select_best_theta must still pick a winner
    # from the un-rejected thetas, using repeatability as the fallback axis.
    thetas = [
        _theta("worse_repeatability", float("nan"), 0.6, 0.99),
        _theta("better_repeatability", float("nan"), 0.9, 0.10),
    ]

    winner = select_best_theta(thetas)

    assert winner["theta_id"] == "better_repeatability"


def test_falls_back_to_quality_score_raw_when_repeatability_also_missing():
    thetas = [
        _theta("lower_quality", float("nan"), float("nan"), 0.10),
        _theta("higher_quality", float("nan"), float("nan"), 0.90),
    ]

    winner = select_best_theta(thetas)

    assert winner["theta_id"] == "higher_quality"


def test_gated_thetas_are_excluded_not_crashing():
    thetas = [
        _theta("gated", float("nan"), float("nan"), 0.99, rejected="density 0.9 outside [0.02, 0.6]"),
    ]

    assert select_best_theta(thetas) is None


def test_all_thetas_gated_returns_none():
    thetas = [
        _theta("gated_a", float("nan"), float("nan"), 0.9, rejected="unequal repeats across subjects"),
        _theta("gated_b", float("nan"), float("nan"), 0.8, rejected="no usable atlas/metric pairs"),
    ]

    assert select_best_theta(thetas) is None


# ---------------------------------------------------------------------------
# _compute_qa_for_theta
# ---------------------------------------------------------------------------


def _write_mrtrix_csv(theta_root, rep, subject, atlas, metric, matrix):
    d = theta_root / f"rep_{rep}" / "results" / atlas
    d.mkdir(parents=True, exist_ok=True)
    labels = [str(i) for i in range(matrix.shape[0])]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        d / f"{subject}_{atlas}.{metric}.connectivity.csv"
    )


def _sym(rng, n=10, scale=100.0):
    upper = np.triu(rng.random((n, n)) * scale, 1)
    return upper + upper.T


def test_compute_qa_for_theta_keeps_quality_score_raw_and_adds_reliability_fields(
    tmp_path, monkeypatch
):
    # quality_score_raw's computation (aggregate_network_measures + MetricOptimizer)
    # is unchanged pre-existing logic; stub it out so this test exercises only the
    # new reliability merge, same isolation Task 2's assemble_ok_result tests use
    # for rep_opt_dfs.
    monkeypatch.setattr(
        "scripts.mrtrix_tune._quality_score_raw_for_theta",
        lambda theta_root: (0.42, {"count": 0.42}),
    )

    rng = np.random.default_rng(1)
    truth_by_subject = {"sub0": _sym(rng), "sub1": _sym(rng)}
    for rep in (1, 2):
        for subject, truth in truth_by_subject.items():
            noisy = truth + _sym(rng, scale=5.0)
            _write_mrtrix_csv(tmp_path, rep, subject, "Schaefer200", "count", noisy)

    qa = _compute_qa_for_theta(tmp_path, {"density_range": [0.02, 1.0]})

    assert qa["quality_score_raw"] == pytest.approx(0.42)
    assert qa["quality_score_raw_by_metric"] == {"count": 0.42}
    assert qa["discriminability"] > 0.5
    assert not math.isnan(qa["repeatability"])
    assert qa["rejected"] == ""


def test_compute_qa_for_theta_keeps_repeatability_when_only_discriminability_is_nan(
    tmp_path, monkeypatch
):
    # Fix round 1: single-subject case. score_combo's row genuinely passes the
    # reliability gates (rejected == "") and has a real repeatability value;
    # discriminability is NaN only because there is one subject.  rank() drops
    # the row (NaN discriminability), but _compute_qa_for_theta must not treat
    # that the same as a real gate failure: repeatability must survive and
    # `rejected` must stay "", not get stamped "no usable atlas/metric pairs".
    monkeypatch.setattr(
        "scripts.mrtrix_tune._quality_score_raw_for_theta",
        lambda theta_root: (0.5, {"count": 0.5}),
    )

    rng = np.random.default_rng(3)
    truth = _sym(rng)
    for rep in (1, 2):
        noisy = truth + _sym(rng, scale=5.0)
        _write_mrtrix_csv(tmp_path, rep, "sub0", "Schaefer200", "count", noisy)

    qa = _compute_qa_for_theta(tmp_path, {"density_range": [0.02, 1.0]})

    assert math.isnan(qa["discriminability"])
    assert not math.isnan(qa["repeatability"])
    assert qa["rejected"] == ""


def test_select_best_theta_picks_a_real_winner_for_single_subject_thetas(tmp_path, monkeypatch):
    # Integration test (review-requested): score real theta_roots through
    # _compute_qa_for_theta -- not hand-built dicts -- for two thetas of one
    # subject, then feed the resulting records into select_best_theta and
    # confirm it returns a real winner instead of None.
    monkeypatch.setattr(
        "scripts.mrtrix_tune._quality_score_raw_for_theta",
        lambda theta_root: (0.5, {"count": 0.5}),
    )

    theta_results = []
    for i, (repeat_scale, quality) in enumerate([(5.0, 0.3), (2.0, 0.7)], start=1):
        theta_root = tmp_path / f"theta_{i:03d}"
        rng = np.random.default_rng(10 + i)
        truth = _sym(rng)
        for rep in (1, 2):
            noisy = truth + _sym(rng, scale=repeat_scale)
            _write_mrtrix_csv(theta_root, rep, "sub0", "Schaefer200", "count", noisy)

        qa = _compute_qa_for_theta(theta_root, {"density_range": [0.02, 1.0]})
        theta_results.append({"theta_id": f"theta_{i:03d}", "quality_score_raw": quality, **qa})

    winner = select_best_theta(theta_results)

    assert winner is not None
    assert winner["rejected"] == ""
    assert not math.isnan(winner["repeatability"])


def test_repeats_loop_varies_mrtrix_rng_seed_per_repeat(tmp_path, capsys):
    bundle = Bundle(
        wm_fod=tmp_path / "wm.mif",
        act_5tt_or_hsvs=None,
        parcellation_dseg=tmp_path / "atlas_dseg.mif",
        parcellation_labels=tmp_path / "atlas_labels.txt",
    )
    cfg = {"reliability": {"repeats": 2}}

    evaluate_theta(
        cfg,
        bundles={"sub-01": bundle},
        atlas="AtlasX",
        out_base=tmp_path / "out",
        theta_id="theta_001",
        theta={},
        nthreads=1,
        enable_act=False,
        enable_sift2=False,
        compute_smallworld=False,
        overwrite=False,
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "MRTRIX_RNG_SEED=1 " in out and "rep_1/sub-01_tractogram.tck" in out
    assert "MRTRIX_RNG_SEED=2 " in out and "rep_2/sub-01_tractogram.tck" in out


def test_evaluate_theta_writes_outputs_for_every_subject_into_shared_theta_dir(
    tmp_path, capsys
):
    # Task 6: sweep/bayes must evaluate multiple subjects per theta, with
    # every subject's repeat-run outputs landing in the SAME rep_<k> dir (so
    # collect_matrices sees them all as one theta). Dry-run + capsys mirrors
    # Task 5's own repeats-loop test since there's no real tckgen binary here.
    bundle_a = Bundle(
        wm_fod=tmp_path / "subA_wm.mif",
        act_5tt_or_hsvs=None,
        parcellation_dseg=tmp_path / "atlas_dseg.mif",
        parcellation_labels=tmp_path / "atlas_labels.txt",
    )
    bundle_b = Bundle(
        wm_fod=tmp_path / "subB_wm.mif",
        act_5tt_or_hsvs=None,
        parcellation_dseg=tmp_path / "atlas_dseg.mif",
        parcellation_labels=tmp_path / "atlas_labels.txt",
    )
    cfg = {"reliability": {"repeats": 1}}

    evaluate_theta(
        cfg,
        bundles={"sub-A": bundle_a, "sub-B": bundle_b},
        atlas="AtlasX",
        out_base=tmp_path / "out",
        theta_id="theta_001",
        theta={},
        nthreads=1,
        enable_act=False,
        enable_sift2=False,
        compute_smallworld=False,
        overwrite=False,
        dry_run=True,
    )

    out = capsys.readouterr().out
    for subject in ("sub-A", "sub-B"):
        assert f"rep_1/{subject}_tractogram.tck" in out
        assert f"rep_1/{subject}_AtlasX.count.connectome_raw.csv" in out


def test_evaluate_theta_single_subject_still_works(tmp_path, capsys):
    # Task 5's single-subject fallback path must still be reachable: a
    # one-entry bundles dict should not error out of the new subject loop.
    bundle = Bundle(
        wm_fod=tmp_path / "wm.mif",
        act_5tt_or_hsvs=None,
        parcellation_dseg=tmp_path / "atlas_dseg.mif",
        parcellation_labels=tmp_path / "atlas_labels.txt",
    )
    cfg = {"reliability": {"repeats": 1}}

    evaluate_theta(
        cfg,
        bundles={"sub-01": bundle},
        atlas="AtlasX",
        out_base=tmp_path / "out",
        theta_id="theta_001",
        theta={},
        nthreads=1,
        enable_act=False,
        enable_sift2=False,
        compute_smallworld=False,
        overwrite=False,
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "rep_1/sub-01_tractogram.tck" in out


def test_evaluate_theta_multi_subject_yields_computable_discriminability(
    tmp_path, monkeypatch
):
    # Regression test for Task 5's structural gap (reproduced by review):
    # discriminability needs >=2 subjects' repeat-run matrices in the SAME
    # theta_dir. This drives evaluate_theta's real subject loop end-to-end
    # (not a hand-built dict) into the real, unmocked _compute_qa_for_theta
    # -> score_combo -> collect_matrices -> discriminability chain, and
    # confirms the result is a real finite number.
    monkeypatch.setattr("scripts.mrtrix_tune._run", lambda *a, **k: None)
    monkeypatch.setattr(
        "scripts.mrtrix_tune._quality_score_raw_for_theta",
        lambda theta_root: (0.5, {"count": 0.5}),
    )
    monkeypatch.setattr("scripts.mrtrix_tune.compute_measures", lambda *a, **k: {})
    monkeypatch.setattr(
        "scripts.mrtrix_tune.write_network_measures_csv", lambda *a, **k: None
    )

    rng = np.random.default_rng(7)
    truths = {"sub-A": _sym(rng), "sub-B": _sym(rng)}
    noise_rng = np.random.default_rng(11)

    def fake_write_connectivity_csv(raw_connectome, labels_path, connectivity_csv):
        # Stand-in for a real tckgen/tck2connectome run: the subject is
        # recoverable from the production filename convention
        # (`<subject>_<atlas>.<metric>...`), exactly what collect_matrices
        # itself relies on.
        subject = connectivity_csv.name.split("_")[0]
        noisy = truths[subject] + _sym(noise_rng, scale=5.0)
        idx = [str(i) for i in range(noisy.shape[0])]
        pd.DataFrame(noisy, index=idx, columns=idx).to_csv(connectivity_csv)

    monkeypatch.setattr(
        "scripts.mrtrix_tune.write_opticonn_connectivity_csv", fake_write_connectivity_csv
    )

    bundle_a = Bundle(
        wm_fod=tmp_path / "a.mif",
        act_5tt_or_hsvs=None,
        parcellation_dseg=tmp_path / "d.mif",
        parcellation_labels=tmp_path / "l.txt",
    )
    bundle_b = Bundle(
        wm_fod=tmp_path / "b.mif",
        act_5tt_or_hsvs=None,
        parcellation_dseg=tmp_path / "d.mif",
        parcellation_labels=tmp_path / "l.txt",
    )
    cfg = {"reliability": {"repeats": 2, "density_range": [0.0, 1.0]}}

    rec = evaluate_theta(
        cfg,
        bundles={"sub-A": bundle_a, "sub-B": bundle_b},
        atlas="AtlasX",
        out_base=tmp_path / "out",
        theta_id="theta_001",
        theta={},
        nthreads=1,
        enable_act=False,
        enable_sift2=False,
        compute_smallworld=False,
        overwrite=False,
        dry_run=False,
    )

    assert rec["rejected"] == ""
    assert not math.isnan(rec["discriminability"])


def test_compute_qa_for_theta_reports_rejection_reason_when_ungated(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.mrtrix_tune._quality_score_raw_for_theta",
        lambda theta_root: (0.1, {}),
    )
    # Only one repeat written -> reliability.score_combo rejects for
    # "fewer than 2 repeats for some subject"; no matrices at all also works,
    # but this exercises the real gate-reason path via score_combo.
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    _write_mrtrix_csv(tmp_path, 1, "sub0", "Schaefer200", "count", matrix)

    qa = _compute_qa_for_theta(tmp_path, {"density_range": [0.02, 1.0]})

    assert math.isnan(qa["discriminability"])
    assert math.isnan(qa["repeatability"])
    assert "fewer than 2 repeats" in qa["rejected"]


def test_evaluate_theta_warns_when_reliability_repeats_is_one(tmp_path, caplog):
    bundle = Bundle(
        wm_fod=tmp_path / "wm.mif",
        act_5tt_or_hsvs=None,
        parcellation_dseg=tmp_path / "atlas_dseg.mif",
        parcellation_labels=tmp_path / "atlas_labels.txt",
    )

    with caplog.at_level("WARNING"):
        evaluate_theta(
            {"reliability": {"repeats": 1}},
            bundles={"sub-01": bundle},
            atlas="AtlasX",
            out_base=tmp_path / "out",
            theta_id="theta_001",
            theta={},
            nthreads=1,
            enable_act=False,
            enable_sift2=False,
            compute_smallworld=False,
            overwrite=False,
            dry_run=True,
        )

    assert "2 repeats" in caplog.text
