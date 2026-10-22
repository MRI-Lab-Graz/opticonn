"""PCA of the global graph measures in every specification of a sweep.

    opticonn pca -i <optimize> [--measures NAME ...] [--variables FILE --predictor COL [--covariate COL]]

Per specification: z-score the measures across subjects, principal components by SVD, PC1 and PC2
(sign fixed on the anchor measure), and, with a predictor, their partial Spearman association with it.
Design: docs/superpowers/specs/2026-10-22-outcome-sensitivity-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts.measures_table import check_one_scan_per_subject, measure_columns, read_measures, read_params
from scripts.outcome_curve import _write_rows, check_subject_overlap, partial_spearman
from scripts.reliability import DEFAULT_GATES, gate_reason
from scripts.variables import load_variables

ANCHOR = "global_efficiency(weighted)"


def pca_row(combo_id, subjects, names, anchor, variables=None, predictor=None, covariates=()) -> dict:
    """PC1 and PC2 of one specification: variance explained, loadings and, with a predictor, associations."""
    row = {"combo_id": combo_id, "n": None, "n_dropped": 0, "reason": "", "evr1": None, "evr2": None,
           "rho1": None, "p1": None, "rho2": None, "p2": None}
    rows = [(s, [m[n] for n in names]) for s, m in subjects.items()]
    complete = [(s, v) for s, v in rows if not any(math.isnan(x) for x in v)]
    row["n_dropped"] = len(rows) - len(complete)
    if len(complete) < 4:
        row["reason"] = "fewer than 4 subjects with every measure"
        return row
    x = np.array([v for _, v in complete])
    constant = [n for n, sd in zip(names, x.std(axis=0, ddof=1)) if sd == 0]
    if constant:
        row["reason"] = "zero spread: " + ", ".join(constant)
        return row
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    _, sing, vt = np.linalg.svd(z, full_matrices=False)
    evr = sing ** 2 / np.sum(sing ** 2)
    row["n"] = len(complete)
    needed = [predictor, *covariates] if predictor else []
    ok = [i for i, (s, _) in enumerate(complete)
          if predictor and s in variables and all(variables[s][c] is not None for c in needed)]
    for k in (1, 2):
        vec = vt[k - 1] * (1.0 if vt[k - 1][names.index(anchor)] >= 0 else -1.0)
        row[f"evr{k}"] = float(evr[k - 1])
        row.update({f"load{k}:{n}": float(v) for n, v in zip(names, vec)})
        if len(ok) >= len(covariates) + 4:
            scores = z @ vec
            fit = partial_spearman([variables[complete[i][0]][predictor] for i in ok], [scores[i] for i in ok],
                                   [[variables[complete[i][0]][c] for i in ok] for c in covariates])
            row[f"rho{k}"], row[f"p{k}"] = fit["rho"], fit["p"]
    return row


def _effects(kept, rho_key, p_key, prefix):
    kept = [r for r in kept if r[rho_key] is not None]
    if not kept:
        return {}
    rho = np.array([r[rho_key] for r in kept])
    return {f"{prefix}_min": float(rho.min()), f"{prefix}_max": float(rho.max()),
            f"{prefix}_n_opposite_sign": int(np.sum(rho * np.median(rho) < 0)),
            f"{prefix}_fraction_p_below_05": float(np.mean([r[p_key] < 0.05 for r in kept]))}


def summarize_pca(rows, names) -> dict:
    """Headline numbers over the specifications that pass the gates and were analysed."""
    kept = [r for r in rows if not r["reference"] and not r["gate"] and r["evr1"] is not None]
    ref = next((r for r in rows if r["reference"] and r["evr1"] is not None), None)
    if not kept:
        raise ValueError("no specification passed the gates and was analysed")
    load = np.array([[r[f"load1:{m}"] for m in names] for r in kept])
    med = np.median(load, axis=0)
    cos = load @ med / (np.linalg.norm(load, axis=1) * np.linalg.norm(med))
    evr = np.array([r["evr1"] for r in kept])
    out = {"n_specifications": len(kept), "evr1_min": float(evr.min()), "evr1_max": float(evr.max()),
           "evr1_median": float(np.median(evr)), "evr2_median": float(np.median([r["evr2"] for r in kept])),
           "loading_cosine_min": float(cos.min()), "loading_cosine_median": float(np.median(cos))}
    out.update(_effects(kept, "rho1", "p1", "rho1"))
    out.update(_effects(kept, "rho2", "p2", "rho2"))
    out.update({"reference_evr1": ref["evr1"] if ref else None, "reference_rho1": ref["rho1"] if ref else None,
                "reference_rho2": ref["rho2"] if ref else None})
    return out


def separating_parameters(rows, params) -> dict:
    """{parameter: {level: (min, max) of PC1 variance explained}} for every swept parameter whose
    levels give non-overlapping ranges: a parameter that alone sorts the specifications into regimes."""
    kept = [r for r in rows if not r["reference"] and not r["gate"] and r["evr1"] is not None]
    out = {}
    for name in sorted({k for p in params.values() for k in p}):
        levels: dict = {}
        for r in kept:
            levels.setdefault(params[r["combo_id"]][name], []).append(r["evr1"])
        ranges = sorted((min(v), max(v), lvl) for lvl, v in levels.items())
        if len(ranges) > 1 and all(len(v) > 1 for v in levels.values()) and all(a[1] < b[0] for a, b in zip(ranges, ranges[1:])):
            out[name] = {lvl: (lo, hi) for lo, hi, lvl in ranges}
    return out


def plot_pca(rows, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kept = sorted((r for r in rows if not r["reference"] and not r["gate"] and r["evr1"] is not None),
                  key=lambda r: r["evr1"])
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(range(len(kept)), [r["evr1"] for r in kept], s=14, color="C0")
    ax.set(xlabel="specification (sorted)", ylabel="variance explained by PC1")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opticonn pca", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    ap.add_argument("--measures-dir", default=None, help="default <input>/measures")
    ap.add_argument("--measures", nargs="+", default=None, help="default: every measure in measures.csv")
    ap.add_argument("--anchor", default=ANCHOR, help="measure whose loading is made positive on every component")
    ap.add_argument("--variables", default=None)
    ap.add_argument("--predictor", default=None)
    ap.add_argument("--covariate", action="append", default=[])
    ap.add_argument("-o", "--output-dir", default=None, help="default <input>/pca")
    a = ap.parse_args(argv)
    mdir = Path(a.measures_dir or Path(a.input) / "measures")
    try:
        measures, meta = read_measures(mdir), read_params(mdir)
        check_one_scan_per_subject(measures)
        names = a.measures or measure_columns(measures)
        unknown = [m for m in [*names, a.anchor] if m not in measure_columns(measures)]
        if a.anchor in unknown:
            raise ValueError(f"the anchor measure {a.anchor} is not in measures.csv; choose another with --anchor")
        if unknown:
            raise ValueError(f"no measure named {', '.join(unknown)}")
        if a.anchor not in names:
            raise ValueError(f"the anchor measure {a.anchor} is not among the measures analysed; choose another with --anchor")
        if bool(a.variables) != bool(a.predictor):
            raise ValueError("--variables and --predictor go together")
        variables = load_variables(a.variables, [a.predictor, *a.covariate])[0] if a.variables else None
        if variables is not None:
            check_subject_overlap(measures, variables)
    except ValueError as err:
        raise SystemExit(f"opticonn pca: {err}")
    by_combo: dict[str, dict] = {}
    for r in measures:
        by_combo.setdefault(r["combo_id"], {})[r["subject"]] = {n: r[n] for n in names}
    gate_inputs = {}
    for r in measures:
        gate_inputs.setdefault(r["combo_id"], []).append((r["density"], r["isolated_fraction"]))
    rows = []
    for cid, subjects in sorted(by_combo.items()):
        row = pca_row(cid, subjects, names, a.anchor, variables, a.predictor, a.covariate)
        dens = float(np.mean([d for d, _ in gate_inputs[cid]]))
        iso = float(np.mean([i for _, i in gate_inputs[cid]]))
        row.update(reference=meta[cid]["reference"],
                   gate=gate_reason(dens, iso, DEFAULT_GATES["density_range"], DEFAULT_GATES["max_isolated_fraction"]))
        rows.append(row)
    out = Path(a.output_dir or Path(a.input) / "pca")
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, out / "pca.csv")
    try:
        summary = summarize_pca(rows, names)
    except ValueError as err:
        raise SystemExit(f"opticonn pca: {err}; see {out / 'pca.csv'} for the reason per specification")
    params = {cid: m["parameters"] for cid, m in meta.items()}
    if any(params.values()):
        summary["separating_parameters"] = {n: {str(k): v for k, v in lv.items()}
                                            for n, lv in separating_parameters(rows, params).items()}
    (out / "pca_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_pca(rows, out / "pca.png")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
