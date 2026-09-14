import numpy as np
import pytest
import scipy.io

from scripts.reliability import (
    collect_matrices,
    discriminability,
    gate_reason,
    graph_stats,
    loo_top1_frequency,
    rank,
    repeatability,
    score_combo,
)

N_NODES = 20


def _sym(rng, scale=100.0):
    upper = np.triu(rng.random((N_NODES, N_NODES)) * scale, 1)
    return upper + upper.T


def _dataset(subject_specific, subjects=4, repeats=2, seed=0):
    """Repeats differ by small noise; subjects differ only if subject_specific."""
    rng = np.random.default_rng(seed)
    shared = _sym(rng)
    data = {}
    for s in range(subjects):
        truth = _sym(rng) if subject_specific else shared
        data[f"sub{s}"] = [truth + _sym(rng, scale=5.0) for _ in range(repeats)]
    return data


def test_subject_specific_connectomes_are_discriminable():
    assert discriminability(_dataset(subject_specific=True)) > 0.95


def test_connectomes_without_subject_signal_score_near_chance():
    # The old score gave near-identical values to very different settings;
    # this is the check that the new one does not.
    assert discriminability(_dataset(subject_specific=False, subjects=10, repeats=3)) < 0.7


def test_identical_connectomes_are_a_tie_not_a_win():
    m = _sym(np.random.default_rng(1))
    assert discriminability({"a": [m, m], "b": [m, m]}) == 0.5


def test_discriminability_undefined_without_two_subjects_with_repeats():
    m = _sym(np.random.default_rng(1))
    assert np.isnan(discriminability({"a": [m, m]}))
    assert np.isnan(discriminability({"a": [m], "b": [m]}))


def test_repeatability_is_one_for_identical_repeats():
    m = _sym(np.random.default_rng(2))
    assert repeatability({"a": [m, m]}) == pytest.approx(1.0)


def test_graph_stats_density_and_isolated_nodes():
    m = np.ones((4, 4))
    np.fill_diagonal(m, 0)
    m[3, :] = 0
    m[:, 3] = 0
    density, isolated = graph_stats({"a": [m]})
    assert density == 0.5  # 3 of 6 possible edges
    assert isolated == 0.25  # node 3


def test_gate_reason():
    assert gate_reason(0.3, 0.0, (0.02, 0.6), 0.1) == ""
    assert "density" in gate_reason(0.9, 0.0, (0.02, 0.6), 0.1)
    assert "isolated" in gate_reason(0.3, 0.5, (0.02, 0.6), 0.1)


def test_rank_drops_rejected_and_nan_then_orders():
    rows = [
        {"id": "low", "discriminability": 0.7, "repeatability": 0.99, "rejected": "", "tract_count": 1},
        {"id": "gated", "discriminability": 1.0, "repeatability": 0.99, "rejected": "density", "tract_count": 1},
        {"id": "tie_worse_rep", "discriminability": 0.9, "repeatability": 0.80, "rejected": "", "tract_count": 1},
        {"id": "best", "discriminability": 0.9, "repeatability": 0.95, "rejected": "", "tract_count": 1},
        {"id": "nan", "discriminability": float("nan"), "repeatability": 0.9, "rejected": "", "tract_count": 1},
    ]
    assert [r["id"] for r in rank(rows)] == ["best", "tie_worse_rep", "low"]


def test_loo_top1_frequency_prefers_the_discriminable_candidate():
    freq = loo_top1_frequency({"signal": _dataset(True), "noise": _dataset(False)})
    assert freq == {"signal": 1.0, "noise": 0.0}


def test_loo_top1_frequency_undefined_below_three_subjects():
    freq = loo_top1_frequency({"x": _dataset(True, subjects=2)})
    assert np.isnan(freq["x"])


def _write_dsi_mat(root, rep, subject, atlas, metric, matrix):
    d = root / f"rep_{rep}" / "01_connectivity" / f"{subject}.gqi_20250101" / "tracks_100k" / "results" / atlas
    d.mkdir(parents=True, exist_ok=True)
    name = f"{subject}.gqi_{atlas}.tt.gz.{atlas}.{metric}..pass.connectivity.mat"
    scipy.io.savemat(str(d / name), {"connectivity": matrix})


def test_collect_and_score_dsi_studio_layout(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        for k, m in enumerate(reps, 1):
            _write_dsi_mat(tmp_path, k, subject, "FreeSurferDKT_Cortical", "count", m)
    got = collect_matrices(tmp_path)
    assert list(got) == [("FreeSurferDKT_Cortical", "count")]
    assert sorted(got[("FreeSurferDKT_Cortical", "count")]) == ["sub0", "sub1", "sub2"]
    [row] = score_combo(tmp_path, {})
    assert row["n_subjects"] == 3 and row["n_repeats"] == 2
    assert row["discriminability"] > 0.95
