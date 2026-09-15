"""Tests for the repeat-run discriminability rewiring in scripts/mrtrix_tune.py.

Mirrors tests/test_cross_validation_repeats.py's pattern: test the pure
selection function and the repeats-aware scoring function against fake data,
not against real tckgen/tck2connectome subprocess execution.
"""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.mrtrix_tune import _compute_qa_for_theta, select_best_theta


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
