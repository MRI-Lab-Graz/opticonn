import numpy as np

from scripts.mrtrix_backend import build_commands, matrix_csv_to_mat
from scripts.reliability import collect_matrices, load_matrix
from scripts.sweep_utils import find_subject_inputs

CFG = {
    "tract_count": 500000,
    "thread_count": 2,
    "tracking_parameters": {
        "algorithm": "iFOD2",
        "cutoff": 0.06,
        "angle": 45,
        "step": 0,
        "min_length": 10,
        "max_length": 0,
        "random_seed": 3,
    },
}


def test_build_commands(tmp_path):
    subj = tmp_path / "sub-01"
    subj.mkdir()
    work = tmp_path / "work"
    tckgen, tck2, env = build_commands(subj, "desikan", CFG, work)

    assert tckgen[:3] == ["tckgen", str(subj / "wmfod.mif"), str(work / "desikan.tck")]
    assert tckgen[tckgen.index("-algorithm") + 1] == "iFOD2"
    assert tckgen[tckgen.index("-select") + 1] == "500000"
    assert tckgen[tckgen.index("-seed_dynamic") + 1] == str(subj / "wmfod.mif")
    assert tckgen[tckgen.index("-cutoff") + 1] == "0.06"
    assert tckgen[tckgen.index("-angle") + 1] == "45"
    assert tckgen[tckgen.index("-minlength") + 1] == "10"
    assert "-maxlength" not in tckgen  # 0 = MRtrix3 default
    assert "-step" not in tckgen
    assert "-act" not in tckgen

    assert tck2[:4] == ["tck2connectome", str(work / "desikan.tck"), str(subj / "desikan.mif"), str(work / "desikan.connectome.csv")]
    assert {"-symmetric", "-zero_diagonal"} <= set(tck2)
    assert env == {"MRTRIX_RNG_SEED": "3"}


def test_act_used_when_5tt_present(tmp_path):
    subj = tmp_path / "sub-01"
    subj.mkdir()
    (subj / "5tt.mif").touch()
    tckgen, _, _ = build_commands(subj, "desikan", CFG, tmp_path)
    assert tckgen[tckgen.index("-act") + 1] == str(subj / "5tt.mif")


def _results_dir(root):
    d = root / "rep_1" / "01_connectivity" / "sub-01" / "results" / "desikan"
    d.mkdir(parents=True)
    return d


M = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=float)


def test_comma_csv_to_mat_is_collectable(tmp_path):
    d = _results_dir(tmp_path)
    np.savetxt(d / "desikan.connectome.csv", M, delimiter=",")
    mat = d / "sub-01_desikan.count..end.connectivity.mat"
    matrix_csv_to_mat(d / "desikan.connectome.csv", mat)
    assert np.array_equal(load_matrix(mat), M)
    got = collect_matrices(tmp_path)
    assert list(got) == [("desikan", "count")]
    assert list(got[("desikan", "count")]) == ["sub-01"]


def test_whitespace_csv_with_comment_header(tmp_path):
    d = _results_dir(tmp_path)
    csv = d / "desikan.connectome.csv"
    np.savetxt(csv, M, header="command_history: tck2connectome ...")
    matrix_csv_to_mat(csv, d / "out.mat")
    assert np.array_equal(load_matrix(d / "out.mat"), M)


def test_find_subject_inputs(tmp_path):
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "sub-01" / "wmfod.mif").touch()
    (tmp_path / "sub-02").mkdir()  # no FOD, not a subject
    (tmp_path / "b.fib.gz").touch()
    (tmp_path / "a.fz").touch()
    assert find_subject_inputs(tmp_path, "mrtrix3") == [tmp_path / "sub-01"]
    assert find_subject_inputs(tmp_path) == [tmp_path / "a.fz", tmp_path / "b.fib.gz"]


import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _mrtrix_subjects(root, n=3):
    for i in range(1, n + 1):
        s = root / f"sub-0{i}"
        s.mkdir()
        (s / "wmfod.mif").touch()
        (s / "desikan.mif").touch()
    return root


def test_config_backend(tmp_path):
    from opticonn import config_backend

    (tmp_path / "sweep.json").write_text('{"backend": "mrtrix3"}')
    (tmp_path / "top3.json").write_text('[{"backend": "mrtrix3"}]')
    (tmp_path / "dsi.json").write_text("{}")
    assert config_backend(tmp_path / "sweep.json") == "mrtrix3"
    assert config_backend(tmp_path / "top3.json") == "mrtrix3"
    assert config_backend(tmp_path / "dsi.json") == "dsi_studio"
    assert config_backend(tmp_path / "missing.json") == "dsi_studio"
    assert config_backend(None) == "dsi_studio"


def test_validate_environment_checks_mrtrix_tools_not_dsi_studio(tmp_path, monkeypatch):
    from opticonn import validate_environment

    monkeypatch.setenv("PATH", str(tmp_path))  # no MRtrix3 binaries here
    monkeypatch.delenv("DSI_STUDIO_CMD", raising=False)
    _, issues = validate_environment("mrtrix3")
    assert any("tckgen" in i for i in issues)
    assert not any("DSI" in i for i in issues)


def test_run_pipeline_dispatches_to_mrtrix_backend_and_skips_aggregation(tmp_path):
    cfg = tmp_path / "mrtrix.json"
    cfg.write_text(json.dumps({"backend": "mrtrix3", "atlases": ["desikan"]}))
    out = subprocess.run(
        [sys.executable, "-m", "scripts.run_pipeline", "--dry-run", "--step", "all",
         "--data-dir", str(tmp_path), "--output", str(tmp_path / "out"), "--extraction-config", str(cfg)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "scripts.mrtrix_backend" in out.stdout
    assert "extract_connectivity_matrices" not in out.stdout
    assert "network_measures" not in out.stdout


def test_select_subjects_picks_mrtrix_subject_folders(tmp_path):
    from scripts.cross_validation_bootstrap_optimizer import select_subjects

    _mrtrix_subjects(tmp_path, n=3)
    (tmp_path / "notes").mkdir()
    picked = select_subjects(tmp_path, "mrtrix3", n_subjects=2, seed=42)
    assert len(picked) == 2 and all((p / "wmfod.mif").exists() for p in picked)
    assert picked == select_subjects(tmp_path, "mrtrix3", n_subjects=2, seed=42)
    assert select_subjects(tmp_path, "mrtrix3", n_subjects=10, seed=42) == sorted(tmp_path.glob("sub-*"))


@pytest.mark.skipif(shutil.which("tckgen") is None, reason="MRtrix3 not installed")
def test_sweep_dry_run_counts_mrtrix_subjects(tmp_path):
    _mrtrix_subjects(tmp_path, n=3)
    out = subprocess.run(
        [sys.executable, "opticonn.py", "sweep", "--config", "configs/mrtrix_quick_sweep.json",
         "--data", str(tmp_path), "--output", str(tmp_path / "results"), "--dry-run"],
        cwd=REPO, capture_output=True, text=True,
    )
    text = out.stdout + out.stderr
    assert out.returncode == 0, text
    assert "Found 3 subjects available" in text
