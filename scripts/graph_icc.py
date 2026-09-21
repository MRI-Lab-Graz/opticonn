"""Reliability of graph measures across a completed OptiConn sweep.

For each candidate parameter set and each global graph measure, a one-way random
ICC(1,1): variance between subjects against variance between tracking repeats
of the same scan. Discriminability saturates at 1.0 for plausible candidates,
but a scalar graph measure compresses the connectome, so tracking noise is no
longer negligible and candidates can separate. Measures may disagree on which
candidate is more reliable, so this is reported per measure and never ranked:
nothing here is consumed by scripts.reliability.rank() or rank_with_fallback().
See docs/superpowers/specs/2026-09-21-cross-sectional-graph-icc-design.md.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist

from scripts.compute_network_measures_from_connectivity import measures_from_matrix
from scripts.variance_decomposition import collect_sweep_matrices

MIN_SUBJECTS_FOR_ICC = 10
MIN_SUBJECTS_TO_REPORT = 3

_COLUMNS = [
    "atlas", "metric", "combo_id", "measure", "n_subjects",
    "icc", "ci_low", "ci_high", "low_confidence", "reason",
]
_OUTPUTS = ("graph_icc.csv", "graph_icc_summary.txt")


def icc_1_1(x: np.ndarray) -> tuple[float, float, float]:
    """One-way random ICC(1,1) and its 95% CI (F distribution). x: subjects x repeats.

    NaN triple when there is no variance at all; 1.0 when repeats agree exactly.
    """
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    row_means = x.mean(axis=1)
    msb = k * ((row_means - x.mean()) ** 2).sum() / (n - 1)
    msw = ((x - row_means[:, None]) ** 2).sum() / (n * (k - 1))
    if msb + (k - 1) * msw == 0:
        return float("nan"), float("nan"), float("nan")
    if msw == 0:
        return 1.0, 1.0, 1.0
    icc = (msb - msw) / (msb + (k - 1) * msw)
    F = msb / msw
    f_low = F / f_dist.ppf(0.975, n - 1, n * (k - 1))
    f_high = F * f_dist.ppf(0.975, n * (k - 1), n - 1)
    return float(icc), float((f_low - 1) / (f_low + k - 1)), float((f_high - 1) / (f_high + k - 1))


def compute_graph_icc(combo_matrices: dict[str, dict[str, list[np.ndarray]]]) -> list[dict]:
    """One row per (combo_id, measure) for one (atlas, metric).

    combo_matrices: {combo_id: {scan_key: [matrix per repeat]}}, as one value of
    collect_sweep_matrices(). Repeats are truncated to the shortest subject so the
    ICC design is balanced.
    """
    rows: list[dict] = []
    for combo_id in sorted(combo_matrices):
        scans = combo_matrices[combo_id]
        usable = {key: reps for key, reps in scans.items() if len(reps) >= 2}
        too_few_repeats = len(scans) - len(usable)
        if not usable:
            logging.warning("graph ICC: %s has no subject with >=2 repeats; skipped", combo_id)
            continue
        k = min(len(reps) for reps in usable.values())
        # ponytail: sequential per-matrix measures (~0.4 s on AAL3, ~14 s on a dense
        # 400-node graph); parallelise with multiprocessing if report time matters.
        per_scan = {key: [measures_from_matrix(m) for m in reps[:k]] for key, reps in usable.items()}
        names = sorted({name for reps in per_scan.values() for r in reps for name in r})
        for name in names:
            series = [[r.get(name, float("nan")) for r in reps] for reps in per_scan.values()]
            clean = [s for s in series if np.all(np.isfinite(s))]
            notes = []
            if too_few_repeats:
                notes.append(f"{too_few_repeats} subject(s) with <2 repeats dropped")
            if len(series) - len(clean):
                notes.append(f"{len(series) - len(clean)} subject(s) with a non-finite value dropped")
            row = {
                "combo_id": combo_id,
                "measure": name,
                "n_subjects": len(clean),
                "icc": None,
                "ci_low": None,
                "ci_high": None,
                "low_confidence": len(clean) < MIN_SUBJECTS_FOR_ICC,
            }
            if len(clean) < MIN_SUBJECTS_TO_REPORT:
                notes.insert(0, f"fewer than {MIN_SUBJECTS_TO_REPORT} subjects ({len(clean)})")
            else:
                icc, low, high = icc_1_1(np.array(clean))
                if np.isnan(icc):
                    notes.insert(0, "no variance across subjects or repeats")
                else:
                    row.update(icc=icc, ci_low=low, ci_high=high)
            row["reason"] = "; ".join(notes)
            rows.append(row)
    return rows


def _summary_lines(atlas: str, metric: str, rows: list[dict]) -> list[str]:
    lines = [f"\n=== {atlas} / {metric} ==="]
    for name in sorted({r["measure"] for r in rows}):
        measure_rows = [r for r in rows if r["measure"] == name]
        scored = [r for r in measure_rows if r["icc"] is not None]
        if not scored:
            lines.append(f"  {name}: not available ({measure_rows[0]['reason'] or 'no candidate scored'})")
            continue
        best = max(scored, key=lambda r: r["icc"])
        flag = " [low confidence]" if best["low_confidence"] else ""
        lines.append(
            f"  {name}: most reliable {best['combo_id']} ICC={best['icc']:.2f} "
            f"[{best['ci_low']:.2f}, {best['ci_high']:.2f}] n={best['n_subjects']}{flag}"
        )
    return lines


def run(sweep_optimize_dir: Path, output_dir: Path) -> list[dict]:
    output_dir = Path(output_dir)
    for name in _OUTPUTS:
        (output_dir / name).unlink(missing_ok=True)
    grouped = collect_sweep_matrices(Path(sweep_optimize_dir))
    if not grouped:
        logging.warning(
            "graph ICC: no */combos/sweep_* directories with connectivity matrices under %s",
            sweep_optimize_dir,
        )
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summary = [
        "Graph-measure reliability: one-way ICC(1,1), subjects vs tracking repeats, 95% CI.",
        "Reported per measure, never ranked. Modularity uses seeded Louvain, so part of its",
        "within-subject variance comes from the algorithm, not from tractography.",
    ]
    with (output_dir / "graph_icc.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for atlas, metric in sorted(grouped):
            rows = [{"atlas": atlas, "metric": metric, **r} for r in compute_graph_icc(grouped[(atlas, metric)])]
            writer.writerows(rows)
            all_rows.extend(rows)
            if rows:
                summary.extend(_summary_lines(atlas, metric, rows))
    (output_dir / "graph_icc_summary.txt").write_text("\n".join(summary) + "\n")
    logging.info("Graph ICC report written to %s", output_dir / "graph_icc.csv")
    return all_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph-measure ICC across a completed OptiConn sweep")
    parser.add_argument("sweep_optimize_dir", help="Path to a sweep's optimize/ directory")
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: <sweep_optimize_dir>/optimization_results)",
    )
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_optimize_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "optimization_results"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(sweep_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
