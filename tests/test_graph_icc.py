import ast
import csv
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from scripts.graph_icc import MIN_SUBJECTS_FOR_ICC, compute_graph_icc, icc_1_1, run

N = 16
MEASURES = {
    "density",
    "global_efficiency(binary)",
    "clustering_coeff_average(binary)",
    "small_worldness(binary)",
    "clustering_coeff_average(weighted)",
    "global_efficiency(weighted)",
    "modularity",
}


def _subject_matrix(rng):
    mask = np.triu(rng.random((N, N)) < 0.4, 1)
    upper = np.where(mask, rng.random((N, N)) * 100, 0.0)
    return upper + upper.T


def _combo(n_subjects, seed=0, repeats=2):
    """{scan_key: [repeat matrices]}: each subject distinct, repeats add small noise on its edges."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_subjects):
        base = _subject_matrix(rng)
        out[f"sub-{i:03d}_ses-1"] = [
            base * (1 + 0.01 * rng.standard_normal(base.shape)) * (base > 0) for _ in range(repeats)
        ]
    return out


def test_icc_matches_hand_computed_value():
    icc, low, high = icc_1_1(np.array([[1.0, 2.0], [3.0, 4.0], [6.0, 8.0]]))
    # MSB = 15.5, MSW = 1.0 -> ICC = 14.5 / 16.5
    assert icc == pytest.approx(29 / 33)
    assert low == pytest.approx(-0.017249, abs=1e-4)
    assert high == pytest.approx(0.996711, abs=1e-4)
    assert low <= icc <= high


def test_icc_near_one_when_repeats_agree_and_subjects_differ():
    subjects = np.arange(20, dtype=float)[:, None] * 10
    x = np.hstack([subjects, subjects + 0.01])
    assert icc_1_1(x)[0] > 0.99


def test_icc_near_zero_when_repeats_vary_as_much_as_subjects():
    x = np.random.default_rng(0).standard_normal((400, 2))
    assert abs(icc_1_1(x)[0]) < 0.15


def test_icc_is_nan_for_zero_variance():
    assert all(math.isnan(v) for v in icc_1_1(np.ones((5, 2))))


def test_compute_graph_icc_reports_every_measure_and_flags_low_confidence():
    rows = compute_graph_icc({"wave1/sweep_0001": _combo(5)})
    assert {r["measure"] for r in rows} == MEASURES
    assert all(r["combo_id"] == "wave1/sweep_0001" for r in rows)
    assert all(r["low_confidence"] for r in rows)  # 5 < MIN_SUBJECTS_FOR_ICC
    density = next(r for r in rows if r["measure"] == "density")
    assert density["n_subjects"] == 5 and density["icc"] is not None


def test_compute_graph_icc_not_low_confidence_at_threshold():
    rows = compute_graph_icc({"c": _combo(MIN_SUBJECTS_FOR_ICC)})
    assert not any(r["low_confidence"] for r in rows)


def test_compute_graph_icc_reports_no_number_below_three_subjects():
    rows = compute_graph_icc({"c": _combo(2)})
    for r in rows:
        assert r["icc"] is None and r["ci_low"] is None and r["ci_high"] is None
        assert r["reason"].startswith("fewer than 3 subjects (2)")


def test_compute_graph_icc_drops_subjects_with_fewer_than_two_repeats():
    combo = _combo(4)
    combo["sub-099_ses-1"] = combo["sub-000_ses-1"][:1]
    rows = compute_graph_icc({"c": combo})
    density = next(r for r in rows if r["measure"] == "density")
    assert density["n_subjects"] == 4
    assert "1 subject(s) with <2 repeats dropped" in density["reason"]


def _write_combo(combo_dir, atlas, metric, reps_by_key):
    for key, reps in reps_by_key.items():
        for rep_idx, matrix in enumerate(reps, start=1):
            d = combo_dir / f"rep_{rep_idx}" / "results" / atlas
            d.mkdir(parents=True, exist_ok=True)
            scipy.io.savemat(
                str(d / f"{key}_{atlas}.tt.gz.{metric}..pass.connectivity.mat"),
                {"connectivity": matrix},
            )


def test_run_writes_csv_and_summary_without_duplicates(tmp_path):
    optimize = tmp_path / "optimize"
    for i, combo_id in enumerate(("sweep_0001", "sweep_0002")):
        _write_combo(optimize / "wave1" / "combos" / combo_id, "AAL3", "count", _combo(4, seed=i))
    out = tmp_path / "optimization_results"

    run(optimize, out)
    run(optimize, out)

    rows = list(csv.DictReader((out / "graph_icc.csv").read_text().splitlines()))
    assert len(rows) == 2 * len(MEASURES)
    assert {r["atlas"] for r in rows} == {"AAL3"} and {r["metric"] for r in rows} == {"count"}
    summary = (out / "graph_icc_summary.txt").read_text()
    assert summary.count("=== AAL3 / count ===") == 1
    assert "Louvain" in summary


def test_run_warns_and_writes_nothing_without_matrices(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        assert run(tmp_path / "missing", tmp_path / "out") == []
    assert not (tmp_path / "out" / "graph_icc.csv").exists()


def test_sweep_hook_runs_graph_icc_in_its_own_try_in_the_two_wave_branch():
    """Both reports run on the two-wave path, each isolated so one failing never
    suppresses the other, and graph ICC never runs on the single-wave path."""
    source = Path(__file__).resolve().parents[1] / "scripts" / "cross_validation_bootstrap_optimizer.py"
    tree = ast.parse(source.read_text())

    def imports(node, module):
        return any(isinstance(c, ast.ImportFrom) and c.module == module for c in ast.walk(node))

    single_wave_if = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Attribute)
        and n.test.attr == "single_wave"
        and any(
            isinstance(c, ast.Constant) and isinstance(c.value, str) and "CROSS-VALIDATION COMPLETED" in c.value
            for c in ast.walk(n)
        )
    )
    assert not any(imports(s, "scripts.graph_icc") for s in single_wave_if.body)
    tries = [n for s in single_wave_if.orelse for n in ast.walk(s) if isinstance(n, ast.Try)]
    icc_tries = [t for t in tries if any(imports(s, "scripts.graph_icc") for s in t.body)]
    vd_tries = [t for t in tries if any(imports(s, "scripts.variance_decomposition") for s in t.body)]
    assert len(icc_tries) == 1 and len(vd_tries) == 1
    assert icc_tries[0] is not vd_tries[0]
