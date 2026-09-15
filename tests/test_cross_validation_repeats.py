"""Tests for repeat-run discriminability selection in the grid/Bayesian sweep."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from scripts.cross_validation_bootstrap_optimizer import assemble_ok_result, select_best_combo


def _combo(name, discriminability, repeatability, quality_score_raw, tract_count=100000, rejected=""):
    return {
        "name": name,
        "discriminability": discriminability,
        "repeatability": repeatability,
        "quality_score_raw": quality_score_raw,
        "tract_count": tract_count,
        "rejected": rejected,
    }


def test_selects_highest_discriminability_even_when_quality_score_raw_is_lower():
    combos = [
        _combo("low_discr_high_quality", discriminability=0.55, repeatability=0.9, quality_score_raw=0.99),
        _combo("high_discr_low_quality", discriminability=0.92, repeatability=0.8, quality_score_raw=0.10),
    ]

    winner = select_best_combo(combos)

    assert winner["name"] == "high_discr_low_quality"


def test_ties_on_discriminability_break_by_repeatability_then_tract_count():
    combos = [
        _combo("tied_low_repeatability", discriminability=0.8, repeatability=0.5, quality_score_raw=0.5, tract_count=200000),
        _combo("tied_high_repeatability", discriminability=0.8, repeatability=0.7, quality_score_raw=0.5, tract_count=200000),
    ]

    winner = select_best_combo(combos)

    assert winner["name"] == "tied_high_repeatability"


def test_combo_with_no_usable_repeats_is_excluded_not_crashing():
    combos = [
        _combo("gated_out", discriminability=float("nan"), repeatability=float("nan"), quality_score_raw=0.99, rejected="fewer than 2 repeats for some subject"),
    ]

    winner = select_best_combo(combos)

    assert winner is None


def test_all_combos_gated_returns_none_not_a_crash():
    combos = [
        _combo("gated_a", discriminability=float("nan"), repeatability=float("nan"), quality_score_raw=0.9, rejected="unequal repeats across subjects"),
        _combo("gated_b", discriminability=math.nan, repeatability=math.nan, quality_score_raw=0.8, rejected="density 0.900 outside [0.02, 0.6]"),
    ]

    assert select_best_combo(combos) is None


def _write_dsi_mat(root, rep, subject, atlas, metric, matrix):
    """Mirrors tests/test_reliability.py's `_write_dsi_mat` legacy DSI Studio layout."""
    d = root / f"rep_{rep}" / "01_connectivity" / f"{subject}.gqi_20250101" / "tracks_100k" / "results" / atlas
    d.mkdir(parents=True, exist_ok=True)
    name = f"{subject}.gqi_{atlas}.tt.gz.{atlas}.{metric}..pass.connectivity.mat"
    scipy.io.savemat(str(d / name), {"connectivity": matrix})


def test_ok_result_records_a_repeat_that_failed_while_another_succeeded(tmp_path):
    # rep_1 "failed" (no .mat files ever written for it -- exactly what a real
    # step01/aggregate/step02 subprocess failure in run_combo leaves behind);
    # rep_2 succeeded and has real tracked matrices.
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    _write_dsi_mat(tmp_path, 2, "sub0", "FreeSurferDKT_Cortical", "count", matrix)

    rep_opt_dfs = [pd.DataFrame({"quality_score_raw": [0.8], "quality_score": [0.9]})]
    partial_failures = ["rep_1: step01_failed: rc=1"]

    result = assemble_ok_result(
        combo_out=tmp_path,
        cfg_path=Path("sweep_0001.json"),
        reliability_cfg={},
        rep_opt_dfs=rep_opt_dfs,
        rep_agg_dfs=[],
        partial_failures=partial_failures,
        first_opt_csv=tmp_path / "rep_2" / "02_optimization" / "optimized_metrics.csv",
        tract_count=100000,
        thread_count=4,
        sweep_meta={},
    )

    # Reliability's own gating (not this function) is what rejects too-thin combos;
    # a genuinely surviving repeat still yields an "ok" result.
    assert result["status"] == "ok"
    assert result["partial_failures"] == partial_failures


def test_ok_result_has_no_partial_failures_when_every_repeat_succeeds(tmp_path):
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    _write_dsi_mat(tmp_path, 1, "sub0", "FreeSurferDKT_Cortical", "count", matrix)

    rep_opt_dfs = [pd.DataFrame({"quality_score_raw": [0.8], "quality_score": [0.9]})]

    result = assemble_ok_result(
        combo_out=tmp_path,
        cfg_path=Path("sweep_0001.json"),
        reliability_cfg={},
        rep_opt_dfs=rep_opt_dfs,
        rep_agg_dfs=[],
        partial_failures=[],
        first_opt_csv=tmp_path / "rep_1" / "02_optimization" / "optimized_metrics.csv",
        tract_count=100000,
        thread_count=4,
        sweep_meta={},
    )

    assert result["status"] == "ok"
    assert result["partial_failures"] == []


def test_select_best_combo_ignores_partial_failures_field():
    combos = [
        _combo("has_failures", discriminability=0.9, repeatability=0.8, quality_score_raw=0.5),
        _combo("no_failures", discriminability=0.6, repeatability=0.8, quality_score_raw=0.5),
    ]
    combos[0]["partial_failures"] = ["rep_1: step01_failed: rc=1"]
    combos[1]["partial_failures"] = []

    winner = select_best_combo(combos)

    # Selection is still purely by discriminability/repeatability/tract_count;
    # the additive partial_failures field must not affect ranking.
    assert winner["name"] == "has_failures"


def test_ok_result_keeps_repeatability_when_only_discriminability_is_nan(tmp_path):
    # Single-subject pool: the reliability gates pass and repeatability is real,
    # but discriminability is structurally NaN.  That is NOT a gate failure --
    # mirrors tests/test_mrtrix_tune.py's equivalent regression test.
    rng = np.random.default_rng(3)
    base = rng.random((12, 12))
    truth = np.triu(base + base.T, 1)
    truth = truth + truth.T
    for rep in (1, 2):
        noise = rng.random((12, 12)) * 0.01
        noisy = truth + np.triu(noise + noise.T, 1) + np.triu(noise + noise.T, 1).T
        _write_dsi_mat(tmp_path, rep, "sub0", "FreeSurferDKT_Cortical", "count", noisy)

    result = assemble_ok_result(
        combo_out=tmp_path,
        cfg_path=Path("sweep_0001.json"),
        reliability_cfg={"density_range": [0.02, 1.0]},
        rep_opt_dfs=[pd.DataFrame({"quality_score_raw": [0.8], "quality_score": [0.9]})],
        rep_agg_dfs=[],
        partial_failures=[],
        first_opt_csv=None,
        tract_count=100000,
        thread_count=4,
        sweep_meta={},
    )

    assert math.isnan(result["discriminability"])
    assert not math.isnan(result["repeatability"])
    assert result["rejected"] == ""


def test_select_best_combo_falls_back_to_repeatability_when_discriminability_is_nan():
    combos = [
        _combo("worse_repeatability", discriminability=float("nan"), repeatability=0.6, quality_score_raw=0.99),
        _combo("better_repeatability", discriminability=float("nan"), repeatability=0.9, quality_score_raw=0.10),
    ]

    winner = select_best_combo(combos)

    assert winner is not None
    assert winner["name"] == "better_repeatability"


def test_select_best_combo_falls_back_to_quality_score_raw_when_repeatability_also_nan():
    combos = [
        _combo("low_quality", discriminability=float("nan"), repeatability=float("nan"), quality_score_raw=0.10),
        _combo("high_quality", discriminability=float("nan"), repeatability=float("nan"), quality_score_raw=0.90),
    ]

    assert select_best_combo(combos)["name"] == "high_quality"
