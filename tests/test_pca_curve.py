# tests/test_pca_curve.py
import json

import numpy as np
import pytest

from scripts.measures_table import write_measures
from scripts.pca_curve import main, pca_row, separating_parameters, summarize_pca
from tests.sweep_fixture import make_sweep

NAMES = ["a", "b", "c", "d", "e", "f", "g"]
ANCHOR = "c"


def _planted(n=60, sign=-1.0, seed=0):
    """Measures a, b, c share one factor that changes with the predictor; the others are noise."""
    rng = np.random.default_rng(seed)
    age = np.linspace(20, 80, n)
    factor = sign * (age - 50) / 30 + 0.3 * rng.standard_normal(n)
    subjects, variables = {}, {}
    for i in range(n):
        row = {m: float(rng.standard_normal()) for m in NAMES}
        for m in "abc":
            row[m] = float(factor[i] + 0.2 * rng.standard_normal())
        subjects[f"sub-{i:02d}"] = row
        variables[f"sub-{i:02d}"] = {"age": float(age[i]), "sex": float(i % 2)}
    return subjects, variables


def test_pc1_picks_up_the_shared_factor_and_its_association_with_the_predictor():
    subjects, variables = _planted(sign=-1.0)
    row = pca_row("s1", subjects, NAMES, ANCHOR, variables, "age", ["sex"])
    assert row["reason"] == "" and row["n"] == 60 and row["n_dropped"] == 0
    assert row["evr1"] > 0.35 and row["load1:c"] > 0
    assert row["rho1"] < -0.8 and row["p1"] < 0.001
    assert abs(row["load1:f"]) < 0.3


def test_the_sign_convention_does_not_depend_on_the_sign_of_the_data():
    subjects, variables = _planted(sign=-1.0)
    a = pca_row("a", subjects, NAMES, ANCHOR, variables, "age")
    subjects, variables = _planted(sign=+1.0)
    b = pca_row("b", subjects, NAMES, ANCHOR, variables, "age")
    assert a["load1:c"] > 0 and b["load1:c"] > 0 and a["rho1"] < 0 < b["rho1"]


def test_without_a_predictor_the_components_are_still_reported():
    subjects, _ = _planted()
    row = pca_row("s1", subjects, NAMES, ANCHOR)
    assert row["evr1"] > 0.35 and row["rho1"] is None


def test_a_subject_with_an_undefined_measure_is_dropped_and_counted():
    subjects, variables = _planted()
    subjects["sub-00"]["f"] = float("nan")
    row = pca_row("s1", subjects, NAMES, ANCHOR, variables, "age")
    assert row["n"] == 59 and row["n_dropped"] == 1


def test_a_constant_measure_is_reported_by_name_and_not_analysed():
    subjects, variables = _planted()
    for m in subjects.values():
        m["e"] = 1.0
    row = pca_row("s1", subjects, NAMES, ANCHOR, variables, "age")
    assert "e" in row["reason"] and row["evr1"] is None


def _prow(i, evr1, rho1, p1, load, ref=False, gate=""):
    r = {"combo_id": f"s{i}", "reference": ref, "gate": gate, "reason": "", "evr1": evr1, "evr2": 0.2,
         "rho1": rho1, "p1": p1, "rho2": 0.0, "p2": 0.5}
    r.update({f"load1:{m}": v for m, v in zip(NAMES, load)})
    return r


def test_summary_reports_variance_association_and_loading_stability():
    base = [0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1]
    rows = [_prow(1, 0.50, 0.30, 0.001, base), _prow(2, 0.60, 0.20, 0.04, base),
            _prow(3, 0.55, -0.10, 0.30, [0.5, -0.4, 0.4, 0.3, -0.3, 0.2, 0.1]), _prow(4, 0.52, 0.10, 0.20, base),
            _prow(5, 0.90, 0.9, 0.001, base, gate="dense"), _prow(6, 0.70, 0.45, 0.001, [0.1] * 7, ref=True)]
    s = summarize_pca(rows, NAMES)
    assert s["n_specifications"] == 4
    assert (s["evr1_min"], s["evr1_max"]) == (0.50, 0.60)
    assert (s["rho1_min"], s["rho1_max"]) == (-0.10, 0.30)
    assert s["rho1_n_opposite_sign"] == 1 and s["rho1_fraction_p_below_05"] == pytest.approx(0.5)
    assert 0 < s["loading_cosine_min"] < 1 and s["loading_cosine_median"] == pytest.approx(1.0, abs=1e-6)
    assert s["reference_evr1"] == 0.70 and s["reference_rho1"] == 0.45


def test_separating_parameters_names_a_parameter_whose_levels_do_not_overlap():
    rows = [{"combo_id": f"c{i}", "reference": False, "gate": "", "evr1": v}
            for i, v in enumerate([0.50, 0.55, 0.60, 0.80, 0.85, 0.90])]
    rows.append({"combo_id": "gated", "reference": False, "gate": "dense", "evr1": 0.1})
    params = {f"c{i}": {"a": 0 if i < 3 else 1, "b": i % 2} for i in range(6)}
    params["gated"] = {"a": 0, "b": 0}
    assert separating_parameters(rows, params) == {"a": {0: (0.50, 0.60), 1: (0.80, 0.90)}}


def test_the_command_runs_on_a_sweep_and_writes_its_outputs(tmp_path):
    specs = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 2), ({"fa_threshold": 0.2}, False, 4),
             ({"fa_threshold": None}, True, 0)]
    make_sweep(tmp_path / "opt", specs, n_subjects=12)
    write_measures(tmp_path / "opt", workers=1)
    assert main(["-i", str(tmp_path / "opt"), "--measures", "density", "global_efficiency(binary)",
                 "global_efficiency(weighted)", "modularity", "--anchor", "global_efficiency(weighted)"]) == 0
    o = tmp_path / "opt" / "pca"
    assert (o / "pca.csv").exists() and (o / "pca.png").stat().st_size > 0
    assert json.loads((o / "pca_summary.json").read_text())["n_specifications"] == 3
    with pytest.raises(SystemExit, match="anchor"):
        main(["-i", str(tmp_path / "opt"), "--anchor", "nope"])


def test_summary_of_nothing_analysed_is_an_error():
    for rows in ([_prow(1, 0.5, 0.1, 0.1, [0.1] * 7, gate="dense")], [_prow(1, None, None, None, [0.1] * 7)]):
        with pytest.raises(ValueError, match="no specification"):
            summarize_pca(rows, NAMES)


def test_an_anchor_outside_the_analysed_measures_is_a_clean_error(tmp_path):
    specs = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 2), ({"fa_threshold": None}, True, 0)]
    make_sweep(tmp_path / "opt", specs, n_subjects=12)
    write_measures(tmp_path / "opt", workers=1)
    with pytest.raises(SystemExit, match="anchor"):
        main(["-i", str(tmp_path / "opt"), "--measures", "density", "modularity"])


def test_when_every_specification_is_gated_the_csv_is_written_and_the_summary_is_not(tmp_path):
    specs = [({"fa_threshold": 0.0}, False, 50), ({"fa_threshold": 0.1}, False, 52), ({"fa_threshold": None}, True, 50)]
    make_sweep(tmp_path / "opt", specs, n_subjects=12)
    write_measures(tmp_path / "opt", workers=1)
    with pytest.raises(SystemExit, match="no specification"):
        main(["-i", str(tmp_path / "opt"), "--measures", "density", "modularity", "--anchor", "modularity"])
    assert (tmp_path / "opt" / "pca" / "pca.csv").exists()
    assert not (tmp_path / "opt" / "pca" / "pca_summary.json").exists()


def test_several_scans_per_subject_are_a_clean_error(tmp_path):
    from tests.sweep_fixture import add_session
    make_sweep(tmp_path / "opt", [({"a": 0}, False, 0), ({"a": 1}, True, 0)], n_subjects=12)
    add_session(write_measures(tmp_path / "opt", workers=1))
    with pytest.raises(SystemExit, match="one scan per subject"):
        main(["-i", str(tmp_path / "opt")])
