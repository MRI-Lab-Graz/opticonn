"""Outcome sensitivity: the same association, estimated in every specification of a sweep.

    opticonn outcome -i <optimize> --variables participants.tsv --predictor age \\
        --covariate sex --measure "global_efficiency(weighted)"

Design: docs/superpowers/specs/2026-10-22-outcome-sensitivity-design.md
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from scipy.stats import t as t_dist


def partial_spearman(x, y, covariates=()) -> dict:
    """Rank correlation of x and y after regressing the ranks of the covariates out of both.

    p from the t distribution with n - 2 - k degrees of freedom, 95% interval by Fisher's z
    with standard error 1/sqrt(n - 3 - k), for k covariates. None where n < k + 4.
    """
    n, k = len(x), len(covariates)
    out = {"rho": None, "p": None, "ci_low": None, "ci_high": None, "n": n}
    if n < k + 4:
        return out
    design = np.column_stack([np.ones(n)] + [rankdata(c) for c in covariates])
    res = [v - design @ np.linalg.lstsq(design, v, rcond=None)[0] for v in (rankdata(x), rankdata(y))]
    rho = np.corrcoef(res)[0, 1]          # numpy float: a perfect fit gives p = 0, not a ZeroDivisionError
    with np.errstate(divide="ignore", invalid="ignore"):
        t = rho * np.sqrt((n - 2 - k) / (1 - rho ** 2))
        half = 1.96 / np.sqrt(n - 3 - k)
        lo, hi = np.tanh(np.arctanh(rho) - half), np.tanh(np.arctanh(rho) + half)
    out.update(rho=float(rho), p=float(2 * t_dist.sf(abs(t), n - 2 - k)), ci_low=float(lo), ci_high=float(hi))
    return out
