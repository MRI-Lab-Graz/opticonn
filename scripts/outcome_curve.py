"""Outcome sensitivity: the same association, estimated in every specification of a sweep.

    opticonn outcome -i <optimize> --variables participants.tsv --predictor age \\
        --covariate sex --measure "global_efficiency(weighted)"

Design: docs/superpowers/specs/2026-10-22-outcome-sensitivity-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as t_dist

from scripts.measures_table import check_one_scan_per_subject, measure_columns, read_measures, read_params
from scripts.reliability import DEFAULT_GATES, gate_reason
from scripts.variables import load_variables

STATS = ("n", "rho", "p", "ci_low", "ci_high")


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
    if min(np.ptp(r) for r in res) < 1e-9:  # constant input: residuals are rounding noise, not spread
        return out
    rho = np.corrcoef(res)[0, 1]          # numpy float: a perfect fit gives p = 0, not a ZeroDivisionError
    with np.errstate(divide="ignore", invalid="ignore"):
        t = rho * np.sqrt((n - 2 - k) / (1 - rho ** 2))
        half = 1.96 / np.sqrt(n - 3 - k)
        lo, hi = np.tanh(np.arctanh(rho) - half), np.tanh(np.arctanh(rho) + half)
    out.update(rho=float(rho), p=float(2 * t_dist.sf(abs(t), n - 2 - k)), ci_low=float(lo), ci_high=float(hi))
    return out


def outcome_rows(measures, meta, variables, predictor, covariates, measure_names, gates=DEFAULT_GATES) -> list[dict]:
    """One row per specification: its gate verdict and, if it passes, the effect of each measure."""
    by_combo: dict[str, list[dict]] = {}
    for r in measures:
        by_combo.setdefault(r["combo_id"], []).append(r)
    needed = [predictor, *covariates]
    rows = []
    for cid in sorted(by_combo):
        subj = by_combo[cid]
        density = float(np.mean([r["density"] for r in subj]))
        isolated = float(np.mean([r["isolated_fraction"] for r in subj]))
        usable = [r for r in subj if r["subject"] in variables
                  and all(variables[r["subject"]][c] is not None for c in needed)]
        row = {"combo_id": cid, "reference": meta[cid]["reference"], "density": density,
               "isolated_fraction": isolated,
               "gate": gate_reason(density, isolated, gates["density_range"], gates["max_isolated_fraction"]),
               "n_subjects": len(subj), "n_missing_variables": len(subj) - len(usable)}
        for name in measure_names:
            row.update({f"{s}:{name}": None for s in STATS})
            ok = [r for r in usable if not math.isnan(r[name])]
            if (not row["gate"] or row["reference"]) and len(ok) >= len(covariates) + 4:
                fit = partial_spearman([variables[r["subject"]][predictor] for r in ok], [r[name] for r in ok],
                                       [[variables[r["subject"]][c] for r in ok] for c in covariates])
                row.update({f"{s}:{name}": fit[s] for s in STATS})
        rows.append(row)
    return rows


def summarize_outcome(rows, measure_names, coding) -> dict:
    """Headline numbers of each measure's curve over the specifications that pass the gates."""
    cands = [r for r in rows if not r["reference"]]
    kept = [r for r in cands if not r["gate"]]
    ref = next((r for r in rows if r["reference"]), None)
    dens_all = np.array([r["density"] for r in kept])
    out = {"n_specifications": len(kept), "n_gated": len(cands) - len(kept), "coding": coding,
           "n_missing_variables_max": max((r["n_missing_variables"] for r in rows), default=0),
           "reference_gate": ref["gate"] if ref else None,
           "reference_density_percentile": float(np.mean(dens_all < ref["density"])) if ref and len(kept) else None,
           "measures": {}}
    for name in measure_names:
        fit = [r for r in kept if r[f"rho:{name}"] is not None and not math.isnan(r[f"rho:{name}"])]
        if not fit:
            out["measures"][name] = {"note": "no specification with an estimable effect"}
            continue
        rho = np.array([r[f"rho:{name}"] for r in fit])
        dens = np.array([r["density"] for r in fit])
        out["measures"][name] = {
            "rho_min": float(rho.min()), "rho_max": float(rho.max()), "rho_median": float(np.median(rho)),
            "n_opposite_sign": int(np.sum(rho * np.median(rho) < 0)),
            "fraction_p_below_05": float(np.mean([r[f"p:{name}"] < 0.05 for r in fit])),
            "rho_vs_density": float(spearmanr(rho, dens)[0]) if len(fit) > 2 and np.ptp(rho) > 0 and np.ptp(dens) > 0 else None,
            "reference_rho": ref[f"rho:{name}"] if ref else None,
            "reference_p": ref[f"p:{name}"] if ref else None}
    return out


def plot_outcome(rows, measure, path) -> None:
    """Left: the effect in every specification, sorted, with intervals. Right: effect against density."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kept = sorted((r for r in rows if not r["reference"] and not r["gate"] and r[f"rho:{measure}"] is not None),
                  key=lambda r: r[f"rho:{measure}"])
    ref = next((r for r in rows if r["reference"]), None)
    rho = np.array([r[f"rho:{measure}"] for r in kept])
    p = np.array([r[f"p:{measure}"] for r in kept])
    x = np.arange(len(kept))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    a.vlines(x, [r[f"ci_low:{measure}"] for r in kept], [r[f"ci_high:{measure}"] for r in kept], color="0.8", lw=1)
    a.scatter(x[p < 0.05], rho[p < 0.05], s=14, color="C0", label="p < 0.05")
    a.scatter(x[p >= 0.05], rho[p >= 0.05], s=14, facecolors="none", edgecolors="C0", label="p >= 0.05")
    b.scatter([r["density"] for r in kept], rho, s=14, color="C0")
    if ref and ref[f"rho:{measure}"] is not None:
        a.axhline(ref[f"rho:{measure}"], color="C3", ls="--", label="reference specification")
        b.scatter([ref["density"]], [ref[f"rho:{measure}"]], marker="D", color="C3", label="reference")
    a.axhline(0, color="k", lw=0.5)
    a.set(xlabel="specification (sorted by effect)", ylabel=f"partial Spearman rho, {measure}")
    b.set(xlabel="mean connectome density", ylabel="rho")
    a.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def check_subject_overlap(measures, variables) -> None:
    ids = sorted({r["subject"] for r in measures})
    if not set(ids) & set(variables):
        raise ValueError("no subject of measures.csv is in the variables table; measures.csv has e.g. "
                         f"{', '.join(ids[:3])}, the variables table has e.g. {', '.join(sorted(variables)[:3])}")


def _write_rows(rows, path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opticonn outcome", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    ap.add_argument("--measures-dir", default=None, help="default <input>/measures")
    ap.add_argument("--variables", required=True, help="table with a participant_id column")
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--covariate", action="append", default=[])
    ap.add_argument("--measure", action="append", required=True, help="repeatable; the first is plotted")
    ap.add_argument("--density-range", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"))
    ap.add_argument("--max-isolated", type=float, default=None)
    ap.add_argument("-o", "--output-dir", default=None, help="default <input>/outcome")
    a = ap.parse_args(argv)
    mdir = Path(a.measures_dir or Path(a.input) / "measures")
    try:
        measures, meta = read_measures(mdir), read_params(mdir)
        check_one_scan_per_subject(measures)
        unknown = [m for m in a.measure if m not in measure_columns(measures)]
        if unknown:
            raise ValueError(f"no measure named {', '.join(unknown)}; measures are {', '.join(measure_columns(measures))}")
        variables, coding = load_variables(a.variables, [a.predictor, *a.covariate])
        check_subject_overlap(measures, variables)
    except (ValueError, OSError) as err:
        raise SystemExit(f"opticonn outcome: {err}")
    gates = {"density_range": a.density_range or DEFAULT_GATES["density_range"],
             "max_isolated_fraction": DEFAULT_GATES["max_isolated_fraction"] if a.max_isolated is None else a.max_isolated}
    rows = outcome_rows(measures, meta, variables, a.predictor, a.covariate, a.measure, gates)
    if not any(not r["gate"] and not r["reference"] for r in rows):
        raise SystemExit("opticonn outcome: no specification passes the gates")
    out = Path(a.output_dir or Path(a.input) / "outcome")
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, out / "outcome.csv")
    summary = summarize_outcome(rows, a.measure, coding)
    (out / "outcome_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_outcome(rows, a.measure[0], out / "outcome.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
