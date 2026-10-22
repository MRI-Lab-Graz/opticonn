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
