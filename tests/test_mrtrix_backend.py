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
