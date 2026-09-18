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


import csv

from scripts.variance_decomposition import headline_text, run, write_decomposition


def test_headline_text_reports_ratio_and_noise_fraction():
    ratios = {"parameter_over_between_session": 1.23, "tracking_noise_over_parameter": 0.025}

    text = headline_text("AAL3", "count", ratios)

    assert "AAL3" in text
    assert "count" in text
    assert "1.2" in text  # 1.23x, allow for rounding


def test_headline_text_handles_missing_ratio():
    text = headline_text("AAL3", "count", {"parameter_over_between_session": None})

    assert "not available" in text.lower()


def test_write_decomposition_creates_csv_with_header_once(tmp_path):
    summaries = {
        "tracking_noise": summarize_stratum({"dissimilarities": [0.01, 0.02], "available": True, "reason": None}),
        "parameter": summarize_stratum({"dissimilarities": [0.1] * 12, "available": True, "reason": None}),
        "between_session": summarize_stratum({"dissimilarities": [], "available": False, "reason": "no repeats"}),
        "between_subject": summarize_stratum({"dissimilarities": [0.3, 0.4, 0.5], "available": True, "reason": None}),
    }
    ratios = compute_ratios(summaries)

    write_decomposition(tmp_path, "AAL3", "count", summaries, ratios)
    write_decomposition(tmp_path, "AAL3", "qa", summaries, ratios)

    csv_path = tmp_path / "variance_decomposition.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8  # 4 strata x 2 metrics
    assert rows[0]["atlas"] == "AAL3"
    assert rows[0]["metric"] == "count"
    assert {r["stratum"] for r in rows[:4]} == {
        "tracking_noise", "parameter", "between_session", "between_subject",
    }

    summary_path = tmp_path / "variance_decomposition_summary.txt"
    assert summary_path.exists()
    text = summary_path.read_text()
    assert "AAL3 / count" in text
    assert "AAL3 / qa" in text


def test_run_end_to_end_writes_output_for_every_atlas_metric(tmp_path):
    optimize_dir = build_sweep_fixture(tmp_path)
    output_dir = tmp_path / "optimization_results"

    results = run(optimize_dir, output_dir)

    assert ("AAL3", "count") in results
    assert results[("AAL3", "count")]["summaries"]["tracking_noise"]["available"] is True
    assert (output_dir / "variance_decomposition.csv").exists()
    assert (output_dir / "variance_decomposition_summary.txt").exists()


def test_run_accepts_a_sweep_shaped_output_dir_directly(tmp_path):
    """The real call site passes output_dir/"optimize" (see cross_validation_
    bootstrap_optimizer.py's own `output_dir` variable) -- confirm run()
    against that exact shape, not just the "optimize_dir already given" case
    tested in Task 6.
    """
    output_dir = tmp_path  # what cross_validation_bootstrap_optimizer.py calls output_dir
    optimize_dir = output_dir / "optimize"
    build_sweep_fixture(output_dir)  # writes into output_dir / "optimize"
    assert optimize_dir.exists()

    results = run(optimize_dir, optimize_dir / "optimization_results")

    assert ("AAL3", "count") in results


def test_variance_decomposition_hook_is_nested_inside_two_wave_branch_only():
    """Static guard against the hook's try/except drifting to be a sibling of
    `if args.single_wave: / else:` (which would make it fire on single-wave
    runs too). Parses cross_validation_bootstrap_optimizer.py's AST and
    asserts the `scripts.variance_decomposition` import lives inside the
    `else:` body of the `if args.single_wave:` statement, not in its `if`
    body and not outside the statement entirely.
    """
    import ast

    source_path = Path(__file__).resolve().parents[1] / "scripts" / "cross_validation_bootstrap_optimizer.py"
    tree = ast.parse(source_path.read_text())

    def imports_variance_decomposition(node) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom) and child.module == "scripts.variance_decomposition":
                return True
        return False

    def mentions_cross_validation_completed(node) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if "CROSS-VALIDATION COMPLETED" in child.value:
                    return True
        return False

    single_wave_ifs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "single_wave"
        and mentions_cross_validation_completed(node)
    ]
    assert single_wave_ifs, "expected the final-summary `if args.single_wave:` statement (with CROSS-VALIDATION COMPLETED in its else branch)"
    if_node = single_wave_ifs[0]

    body_has_hook = any(imports_variance_decomposition(stmt) for stmt in if_node.body)
    orelse_has_hook = any(imports_variance_decomposition(stmt) for stmt in if_node.orelse)

    assert not body_has_hook, "variance decomposition hook must not run on the args.single_wave path"
    assert orelse_has_hook, "variance decomposition hook must run on the two-wave (else) path"
