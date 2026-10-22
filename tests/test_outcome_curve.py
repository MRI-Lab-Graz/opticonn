import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from scripts.outcome_curve import partial_spearman


def _inverse_formula(x, y, covariates):
    """Partial correlation from the inverse correlation matrix of the ranks (independent route)."""
    cols = [rankdata(x), rankdata(y)] + [rankdata(c) for c in covariates]
    p = np.linalg.inv(np.corrcoef(cols))
    return -p[0, 1] / np.sqrt(p[0, 0] * p[1, 1])


X = [21, 25, 30, 34, 41, 47, 52, 60, 66, 71, 38, 44]
Y = [3.1, 2.9, 3.4, 2.8, 2.6, 2.9, 2.2, 2.4, 2.0, 2.1, 3.0, 2.5]
Z1 = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
Z2 = [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 10, 11]


def test_no_covariate_is_the_plain_spearman_with_its_p_value():
    got = partial_spearman(X, Y)
    ref = spearmanr(X, Y)
    assert got["rho"] == pytest.approx(ref.statistic) and got["p"] == pytest.approx(ref.pvalue)
    assert got["n"] == 12


def test_one_and_two_covariates_match_the_inverse_matrix_formula():
    assert partial_spearman(X, Y, [Z1])["rho"] == pytest.approx(_inverse_formula(X, Y, [Z1]))
    assert partial_spearman(X, Y, [Z1, Z2])["rho"] == pytest.approx(_inverse_formula(X, Y, [Z1, Z2]))


def test_the_interval_brackets_the_estimate_and_p_uses_n_minus_2_minus_k_df():
    from scipy.stats import t
    got = partial_spearman(X, Y, [Z1, Z2])
    assert got["ci_low"] < got["rho"] < got["ci_high"]
    df = 12 - 2 - 2
    tval = got["rho"] * np.sqrt(df / (1 - got["rho"] ** 2))
    assert got["p"] == pytest.approx(2 * t.sf(abs(tval), df))


def test_a_perfect_fit_gives_p_zero_instead_of_crashing():
    got = partial_spearman(list(range(10)), list(range(10)), [[0, 1] * 5])
    assert got["rho"] == pytest.approx(1.0) and got["p"] == 0.0


def test_too_few_subjects_gives_no_estimate():
    got = partial_spearman([1, 2, 3, 4], [1, 2, 4, 3], [[0, 1, 0, 1]])
    assert got["n"] == 4 and got["rho"] is None and got["p"] is None


def test_constant_y_with_a_covariate_gives_no_estimate():
    got = partial_spearman([1, 2, 3, 4, 5, 6], [1] * 6, [[1, 2, 3, 4, 5, 7]])
    assert got["n"] == 6
    assert got["rho"] is None and got["p"] is None and got["ci_low"] is None and got["ci_high"] is None


def test_constant_y_without_a_covariate_gives_no_estimate_and_no_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = partial_spearman([1, 2, 3, 4, 5, 6], [1] * 6)
    assert got["rho"] is None and got["p"] is None


def test_constant_x_with_a_covariate_gives_no_estimate():
    got = partial_spearman([1] * 6, [1, 2, 3, 4, 5, 6], [[1, 2, 3, 4, 5, 7]])
    assert got["rho"] is None and got["p"] is None


import json

from scripts.measures_table import read_measures, read_params, write_measures
from scripts.outcome_curve import main, outcome_rows, summarize_outcome
from tests.sweep_fixture import make_sweep

SPECS = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 0),
         ({"fa_threshold": 0.2}, False, 50), ({"fa_threshold": None}, True, 0)]


def _setup(tmp_path, n=15):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=n)
    out = write_measures(tmp_path / "opt", workers=1)
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\n" + "".join(
            f"{i + 1:02d}\t{20 + 3 * i}\t{'M' if i % 2 else 'F'}\n" for i in range(n)))
    return tmp_path / "opt", out


def _rows(tmp_path, **kw):
    from scripts.variables import load_variables
    opt, out = _setup(tmp_path)
    variables, coding = load_variables(tmp_path / "participants.tsv", ["age", "sex"])
    rows = outcome_rows(read_measures(out), read_params(out), variables, "age", ["sex"], ["density"], **kw)
    return rows, coding


def test_each_specification_gets_a_gate_verdict_and_the_effect_if_it_passes(tmp_path):
    rows, _ = _rows(tmp_path)
    by = {r["combo_id"]: r for r in rows}
    assert by["sweep_0001"]["gate"] == "" and by["sweep_0001"]["rho:density"] == pytest.approx(1.0)
    assert by["sweep_0001"]["n:density"] == 15 and by["sweep_0001"]["n_missing_variables"] == 0
    assert "density" in by["sweep_0003"]["gate"] and by["sweep_0003"]["rho:density"] is None
    assert by["sweep_0004"]["reference"] is True and by["sweep_0004"]["rho:density"] == pytest.approx(1.0)


def test_a_subject_without_the_predictor_is_dropped_and_counted(tmp_path):
    from scripts.variables import load_variables
    opt, out = _setup(tmp_path)
    (tmp_path / "participants.tsv").write_text(
        (tmp_path / "participants.tsv").read_text().replace("\t20\t", "\tn/a\t", 1))
    variables, _ = load_variables(tmp_path / "participants.tsv", ["age", "sex"])
    rows = outcome_rows(read_measures(out), read_params(out), variables, "age", ["sex"], ["density"])
    first = next(r for r in rows if r["combo_id"] == "sweep_0001")
    assert first["n_missing_variables"] == 1 and first["n:density"] == 14


def test_the_summary_describes_the_curve_the_gates_and_the_reference(tmp_path):
    rows, coding = _rows(tmp_path)
    s = summarize_outcome(rows, ["density"], coding)
    assert s["n_specifications"] == 2 and s["n_gated"] == 1
    assert s["coding"] == {"sex": {"0": "F", "1": "M"}}
    m = s["measures"]["density"]
    assert m["rho_min"] == pytest.approx(1.0) and m["n_opposite_sign"] == 0
    assert m["fraction_p_below_05"] == 1.0 and m["reference_rho"] == pytest.approx(1.0)
    assert s["reference_gate"] == "" and s["reference_density_percentile"] is not None


def test_the_command_writes_the_table_summary_and_figure(tmp_path):
    opt, out = _setup(tmp_path)
    assert main(["-i", str(opt), "--variables", str(tmp_path / "participants.tsv"), "--predictor", "age",
                 "--covariate", "sex", "--measure", "density"]) == 0
    o = opt / "outcome"
    assert (o / "outcome.csv").exists() and (o / "outcome.png").stat().st_size > 0
    assert json.loads((o / "outcome_summary.json").read_text())["n_specifications"] == 2


def test_missing_measures_table_and_unknown_measure_are_clear_errors(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=5)
    (tmp_path / "v.tsv").write_text("participant_id\tage\n01\t20\n")
    with pytest.raises(SystemExit, match="opticonn measures"):
        main(["-i", str(tmp_path / "opt"), "--variables", str(tmp_path / "v.tsv"), "--predictor", "age",
              "--measure", "density"])
    opt, _ = _setup(tmp_path / "x")
    with pytest.raises(SystemExit, match="no measure named nope"):
        main(["-i", str(opt), "--variables", str(tmp_path / "x" / "participants.tsv"), "--predictor", "age",
              "--measure", "nope"])


def _summary_for(tmp_path, specs, n=15):
    from scripts.variables import load_variables
    make_sweep(tmp_path / "opt", specs, n_subjects=n)
    out = write_measures(tmp_path / "opt", workers=1)
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\n" + "".join(
            f"{i + 1:02d}\t{20 + 3 * i}\t{'M' if i % 2 else 'F'}\n" for i in range(n)))
    variables, coding = load_variables(tmp_path / "participants.tsv", ["age", "sex"])
    rows = outcome_rows(read_measures(out), read_params(out), variables, "age", ["sex"], ["density"])
    return rows, summarize_outcome(rows, ["density"], coding)


def test_a_constant_curve_gives_a_null_rho_vs_density_that_is_valid_json(tmp_path):
    import warnings
    same = [({"fa_threshold": t}, False, 0) for t in (0.0, 0.1, 0.2)] + [({"fa_threshold": None}, True, 0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, s = _summary_for(tmp_path, same)
    assert s["measures"]["density"]["rho_vs_density"] is None
    json.dumps(s, allow_nan=False)


def test_a_gated_reference_is_still_estimated_and_reported(tmp_path):
    specs = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 50),
             ({"fa_threshold": None}, True, 50)]
    rows, s = _summary_for(tmp_path, specs)
    by = {r["combo_id"]: r for r in rows}
    assert by["sweep_0003"]["gate"] != "" and by["sweep_0003"]["rho:density"] == pytest.approx(1.0)
    assert by["sweep_0002"]["gate"] != "" and by["sweep_0002"]["rho:density"] is None
    assert s["reference_gate"] != "" and s["measures"]["density"]["reference_rho"] == pytest.approx(1.0)


def test_several_scans_per_subject_are_a_clean_error(tmp_path):
    from tests.sweep_fixture import add_session
    opt, out = _setup(tmp_path)
    add_session(out)
    with pytest.raises(SystemExit, match="one scan per subject"):
        main(["-i", str(opt), "--variables", str(tmp_path / "participants.tsv"), "--predictor", "age",
              "--measure", "density"])
