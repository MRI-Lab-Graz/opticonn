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
