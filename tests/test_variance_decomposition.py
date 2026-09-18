from pathlib import Path

import numpy as np
import scipy.io

from scripts.variance_decomposition import collect_sweep_matrices

N_NODES = 10


def _make_matrix(rng, scale=100.0):
    upper = np.triu(rng.random((N_NODES, N_NODES)) * scale, 1)
    return upper + upper.T


def _write_combo(
    combo_dir: Path,
    atlas: str,
    metric: str,
    subj_sess_reps: dict[str, list[np.ndarray]],
) -> None:
    """subj_sess_reps: {"sub-001_ses-1": [matrix_rep1, matrix_rep2, ...]}"""
    for subj_sess, reps in subj_sess_reps.items():
        for rep_idx, matrix in enumerate(reps, start=1):
            rep_dir = combo_dir / f"rep_{rep_idx}" / "results" / atlas
            rep_dir.mkdir(parents=True, exist_ok=True)
            out = rep_dir / f"{subj_sess}_{atlas}.tt.gz.{metric}..pass.connectivity.mat"
            scipy.io.savemat(str(out), {"connectivity": matrix})


def build_sweep_fixture(tmp_path: Path) -> Path:
    """A 2-wave, 2-combo sweep: wave1/sweep_0001, wave1/sweep_0002.
    3 subjects, 2 of which (sub-001, sub-002) have 2 sessions each; sub-003
    has one session; 2 reps per subj_sess per combo.
    """
    rng = np.random.default_rng(0)
    optimize_dir = tmp_path / "optimize"

    subj_sess_keys = [
        "sub-001_ses-1",
        "sub-001_ses-2",
        "sub-002_ses-1",
        "sub-002_ses-2",
        "sub-003_ses-1",
    ]

    for combo_id in ("sweep_0001", "sweep_0002"):
        combo_dir = optimize_dir / "wave1" / "combos" / combo_id
        reps = {key: [_make_matrix(rng), _make_matrix(rng)] for key in subj_sess_keys}
        _write_combo(combo_dir, "AAL3", "count", reps)

    return optimize_dir


def test_collect_sweep_matrices_groups_by_atlas_metric_combo_subject(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)

    grouped = collect_sweep_matrices(optimize_dir)

    assert ("AAL3", "count") in grouped
    combo_matrices = grouped[("AAL3", "count")]
    assert set(combo_matrices) == {"wave1/sweep_0001", "wave1/sweep_0002"}
    one_combo = combo_matrices["wave1/sweep_0001"]
    assert set(one_combo) == {
        "sub-001_ses-1",
        "sub-001_ses-2",
        "sub-002_ses-1",
        "sub-002_ses-2",
        "sub-003_ses-1",
    }
    assert len(one_combo["sub-001_ses-1"]) == 2  # 2 reps


def test_collect_sweep_matrices_empty_tree_returns_empty_dict(tmp_path):
    optimize_dir = tmp_path / "optimize"
    optimize_dir.mkdir()

    assert collect_sweep_matrices(optimize_dir) == {}


from scripts.variance_decomposition import compute_strata


def test_compute_strata_tracking_noise_is_within_subject_within_combo(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    grouped = collect_sweep_matrices(optimize_dir)
    combo_matrices = grouped[("AAL3", "count")]

    strata = compute_strata(combo_matrices)

    # 5 subj_sess keys x 2 combos x 1 pair (2 reps -> C(2,2)=1 pair) = 10
    assert strata["tracking_noise"]["available"] is True
    assert len(strata["tracking_noise"]["dissimilarities"]) == 10


def test_compute_strata_parameter_compares_same_subject_across_combos(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    grouped = collect_sweep_matrices(optimize_dir)
    combo_matrices = grouped[("AAL3", "count")]

    strata = compute_strata(combo_matrices)

    # 5 subj_sess keys, each present in both combos -> C(2,2)=1 pair each = 5
    assert strata["parameter"]["available"] is True
    assert len(strata["parameter"]["dissimilarities"]) == 5


def test_compute_strata_between_session_only_pairs_same_subject(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    grouped = collect_sweep_matrices(optimize_dir)
    combo_matrices = grouped[("AAL3", "count")]

    strata = compute_strata(combo_matrices)

    # sub-001 (2 sessions) + sub-002 (2 sessions) -> 1 pair each = 2 per combo,
    # x 2 combos = 4. sub-003 has 1 session, contributes nothing.
    assert strata["between_session"]["available"] is True
    assert len(strata["between_session"]["dissimilarities"]) == 4


def test_compute_strata_between_subject_excludes_same_subject_pairs(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    grouped = collect_sweep_matrices(optimize_dir)
    combo_matrices = grouped[("AAL3", "count")]

    strata = compute_strata(combo_matrices)

    # 5 keys -> C(5,2)=10 raw pairs per combo, minus the 2 same-subject pairs
    # (sub-001 ses1/ses2, sub-002 ses1/ses2) = 8 per combo, x 2 combos = 16
    assert strata["between_subject"]["available"] is True
    assert len(strata["between_subject"]["dissimilarities"]) == 16


def test_compute_strata_between_session_unavailable_for_single_session_cohort(tmp_path):
    rng = np.random.default_rng(1)
    optimize_dir = tmp_path / "optimize"
    combo_dir = optimize_dir / "wave1" / "combos" / "sweep_0001"
    reps = {
        "sub-001_ses-1": [_make_matrix(rng), _make_matrix(rng)],
        "sub-002_ses-1": [_make_matrix(rng), _make_matrix(rng)],
    }
    _write_combo(combo_dir, "AAL3", "count", reps)

    grouped = collect_sweep_matrices(optimize_dir)
    strata = compute_strata(grouped[("AAL3", "count")])

    assert strata["between_session"]["available"] is False
    assert strata["between_session"]["dissimilarities"] == []
    assert strata["between_session"]["reason"] is not None


def test_compute_strata_parameter_unavailable_for_single_combo(tmp_path):
    rng = np.random.default_rng(2)
    optimize_dir = tmp_path / "optimize"
    combo_dir = optimize_dir / "wave1" / "combos" / "sweep_0001"
    reps = {"sub-001_ses-1": [_make_matrix(rng), _make_matrix(rng)]}
    _write_combo(combo_dir, "AAL3", "count", reps)

    grouped = collect_sweep_matrices(optimize_dir)
    strata = compute_strata(grouped[("AAL3", "count")])

    assert strata["parameter"]["available"] is False


import pytest

from scripts.variance_decomposition import (
    MIN_PAIRS_FOR_CONFIDENCE,
    compute_ratios,
    summarize_stratum,
)


def test_summarize_stratum_computes_distribution_stats():
    entry = {"dissimilarities": [0.1, 0.2, 0.3, 0.4], "available": True, "reason": None}

    summary = summarize_stratum(entry)

    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(0.25)
    assert summary["median"] == pytest.approx(0.25)
    assert summary["available"] is True
    assert summary["low_confidence"] is True  # 4 < MIN_PAIRS_FOR_CONFIDENCE


def test_summarize_stratum_not_low_confidence_above_threshold():
    entry = {
        "dissimilarities": [0.1] * MIN_PAIRS_FOR_CONFIDENCE,
        "available": True,
        "reason": None,
    }

    summary = summarize_stratum(entry)

    assert summary["low_confidence"] is False


def test_summarize_stratum_unavailable_passes_through_reason():
    entry = {"dissimilarities": [], "available": False, "reason": "single-session cohort"}

    summary = summarize_stratum(entry)

    assert summary["available"] is False
    assert summary["n"] == 0
    assert summary["mean"] is None
    assert summary["reason"] == "single-session cohort"


def test_compute_ratios_divides_means():
    summaries = {
        "parameter": {"available": True, "mean": 0.2, "n": 20},
        "between_session": {"available": True, "mean": 0.1, "n": 20},
        "tracking_noise": {"available": True, "mean": 0.02, "n": 20},
        "between_subject": {"available": True, "mean": 0.4, "n": 20},
    }

    ratios = compute_ratios(summaries)

    assert ratios["parameter_over_between_session"] == pytest.approx(2.0)
    assert ratios["tracking_noise_over_parameter"] == pytest.approx(0.1)


def test_compute_ratios_none_when_a_stratum_unavailable():
    summaries = {
        "parameter": {"available": True, "mean": 0.2, "n": 20},
        "between_session": {"available": False, "mean": None, "n": 0},
        "tracking_noise": {"available": True, "mean": 0.02, "n": 20},
        "between_subject": {"available": True, "mean": 0.4, "n": 20},
    }

    ratios = compute_ratios(summaries)

    assert ratios["parameter_over_between_session"] is None
    assert ratios["tracking_noise_over_parameter"] is not None
