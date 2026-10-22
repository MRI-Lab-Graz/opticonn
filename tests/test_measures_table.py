# tests/test_measures_table.py
import csv
import json
import math

import numpy as np
import pytest

from scripts.measures_table import (EXTRA, extra_measures, find_wave, matrix_row, measure_columns,
                                    read_measures, read_params, write_measures)
from tests.sweep_fixture import make_sweep

SPECS = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 0),
         ({"fa_threshold": 0.2}, False, 60),          # gated later: density above 0.6
         ({"fa_threshold": None}, True, 0)]


def _path4(w=1.0):
    m = np.zeros((4, 4))
    for i in range(3):
        m[i, i + 1] = m[i + 1, i] = w
    return m


def test_a_path_graph_has_hand_computable_extra_measures():
    got = extra_measures(_path4(2.0))
    assert set(got) == set(EXTRA)
    assert got["local_efficiency(binary)"] == 0.0
    assert got["mean_strength(weighted)"] == pytest.approx(3.0)
    assert got["characteristic_path_length(binary)"] == pytest.approx(10 / 6)
    assert got["mean_betweenness(binary)"] == pytest.approx(1 / 3)
    assert got["degree_assortativity(binary)"] == pytest.approx(-0.5)


def test_a_complete_graph_is_maximal_locally_and_has_undefined_assortativity():
    got = extra_measures(np.ones((5, 5)) - np.eye(5))
    assert got["local_efficiency(binary)"] == pytest.approx(1.0)
    assert got["characteristic_path_length(binary)"] == pytest.approx(1.0)
    assert got["mean_betweenness(binary)"] == 0.0
    assert math.isnan(got["degree_assortativity(binary)"])


def test_path_length_uses_the_largest_connected_component():
    m = np.zeros((5, 5))
    m[0, 1] = m[1, 0] = m[1, 2] = m[2, 1] = 1.0
    assert extra_measures(m)["characteristic_path_length(binary)"] == pytest.approx(4 / 3)


def test_matrix_row_has_the_standard_measures_and_the_isolated_fraction():
    row = matrix_row(_path4(), extended=False)
    assert "density" in row and "global_efficiency(weighted)" in row
    assert not set(EXTRA) & set(row)
    m = np.zeros((4, 4)); m[0, 1] = m[1, 0] = m[1, 2] = m[2, 1] = 1.0     # node 3 isolated
    assert matrix_row(m, extended=False)["isolated_fraction"] == pytest.approx(0.25)
    assert set(EXTRA) <= set(matrix_row(_path4(), extended=True))


def test_find_wave_handles_one_wave_two_waves_and_a_named_wave(tmp_path):
    make_sweep(tmp_path, SPECS, n_subjects=5, wave="wave1")
    assert find_wave(tmp_path).name == "wave1"
    make_sweep(tmp_path, SPECS, n_subjects=5, wave="wave2")
    with pytest.raises(ValueError, match="2 waves"):
        find_wave(tmp_path)
    assert find_wave(tmp_path, "wave2").name == "wave2"


def test_write_measures_makes_one_row_per_specification_and_subject(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    out = write_measures(tmp_path / "opt", workers=1)
    rows = read_measures(out)
    assert len(rows) == 4 * 5
    assert {r["subject"] for r in rows} == {f"sub-0{i}" for i in range(1, 6)}
    assert "sub-01_ses-1_dwi" in {r["scan"] for r in rows} and isinstance(rows[0]["density"], float)
    assert "isolated_fraction" not in measure_columns(rows) and "density" in measure_columns(rows)
    side = json.loads((out / "measures.json").read_text())
    assert side["atlas"] == "AAL3" and side["metric"] == "count" and side["repeat"] == 1
    assert side["wave"] == "wave1" and side["extended"] is False


def test_params_records_the_reference_and_the_swept_parameters(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    params = read_params(write_measures(tmp_path / "opt", workers=1))
    assert params["sweep_0004"]["reference"] is True and params["sweep_0001"]["reference"] is False
    assert params["sweep_0002"]["parameters"] == {"fa_threshold": "0.1"}


def test_extended_adds_the_five_measures_to_the_table(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    rows = read_measures(write_measures(tmp_path / "opt", extended=True, workers=1))
    assert set(EXTRA) <= set(measure_columns(rows))


def test_an_ambiguous_metric_is_an_error_until_chosen(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5, metrics=("count", "qa"))
    with pytest.raises(ValueError, match="more than one.*count.*qa"):
        write_measures(tmp_path / "opt", workers=1)
    out = write_measures(tmp_path / "opt", metric="qa", workers=1)
    assert json.loads((out / "measures.json").read_text())["metric"] == "qa"


def test_two_scans_of_one_subject_in_one_specification_are_an_error():
    from scripts.measures_table import check_one_scan_per_subject
    rows = [{"combo_id": "sweep_0001", "subject": "sub-01", "scan": s} for s in ("sub-01_ses-1", "sub-01_ses-2")]
    with pytest.raises(ValueError, match="sub-01.*sweep_0001.*sub-01_ses-1.*sub-01_ses-2.*one scan per subject"):
        check_one_scan_per_subject(rows)
    check_one_scan_per_subject(rows[:1])


def _drop_matrices(opt, index):
    import shutil
    shutil.rmtree(opt / "wave1" / "combos" / f"sweep_{index:04d}" / "rep_1")


def test_a_specification_without_matrices_is_absent_and_counted(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    _drop_matrices(tmp_path / "opt", 2)
    out = write_measures(tmp_path / "opt", workers=1)
    rows = read_measures(out)
    assert {r["combo_id"] for r in rows} == {"sweep_0001", "sweep_0003", "sweep_0004"}
    assert json.loads((out / "measures.json").read_text())["n_specifications_without_matrices"] == 1


def test_atlas_and_metric_are_found_when_the_first_specification_is_empty(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    _drop_matrices(tmp_path / "opt", 1)
    out = write_measures(tmp_path / "opt", workers=1)
    assert json.loads((out / "measures.json").read_text())["atlas"] == "AAL3"
    assert len(read_measures(out)) == 3 * 5


def test_a_sweep_with_no_matrices_at_all_is_an_error(tmp_path):
    make_sweep(tmp_path / "opt", SPECS[:2], n_subjects=5)
    _drop_matrices(tmp_path / "opt", 1)
    _drop_matrices(tmp_path / "opt", 2)
    with pytest.raises(ValueError, match="no specification has connectivity matrices"):
        write_measures(tmp_path / "opt", workers=1)
