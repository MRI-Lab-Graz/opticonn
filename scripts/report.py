# scripts/report.py
"""Sweep-level sensitivity report: is the leader determined, how fragile is the subject ordering,
how many independent axes do the graph measures carry, how far are the specifications from the defaults.

    opticonn report -i <optimize> [--atlas A] [--metric M] [-o DIR] [--workers N] [--recompute]

Writes <optimize>/report/report.json and report.csv. A statistic the sweep cannot support (one wave, one
repeat, no reference) is skipped with a stated reason. Design:
docs/superpowers/specs/2026-10-25-sensitivity-report-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts import report_stats as stats
from scripts.measures_table import (find_waves, measure_columns, read_measures, read_params, tables_current,
                                    write_measures, count_files)
from scripts.reliability import collect_matrices, edge_vector


def _num(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _diagnostics(wave_dir: Path) -> list[dict] | None:
    path = Path(wave_dir) / "combo_diagnostics.csv"
    if not path.exists():
        return None
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _clean(obj):
    """NaN and infinity become None, numpy scalars become Python ones: the JSON stays strict."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _ensure_table(optimize_dir, out, repeat, atlas, metric, n_specs, n_files, workers, recompute):
    d = out / "tables" / f"repeat{repeat}"
    if recompute or not tables_current(d, atlas, metric, repeat, "all", n_specs, n_files=n_files, extended=False):
        write_measures(optimize_dir, out_dir=d, atlas=atlas, metric=metric, repeat=repeat, workers=workers, wave="all")
    return d


def build_report(optimize_dir, out_dir=None, atlas=None, metric=None, workers=8, recompute=False) -> dict:
    optimize_dir = Path(optimize_dir)
    out = Path(out_dir) if out_dir else optimize_dir / "report"
    waves = find_waves(optimize_dir, "all")
    scoped = len(waves) > 1
    n_specs = sum(len(list((w / "combos").glob("sweep_*"))) for w in waves)
    n_files = count_files(waves)
    notes = []

    t1 = _ensure_table(optimize_dir, out, 1, atlas, metric, n_specs, n_files, workers, recompute)
    rows1 = read_measures(t1)
    params = read_params(t1)
    side = json.loads((t1 / "measures.json").read_text())
    t2 = rows2 = None
    if any(next((w / "combos").glob("sweep_*/rep_2"), None) for w in waves):
        try:
            t = _ensure_table(optimize_dir, out, 2, side["atlas"], side["metric"], n_specs, n_files, workers, recompute)
            rows2, t2 = read_measures(t), t
        except ValueError as err:
            if "has repeat" not in str(err):
                raise
    if t2 is None:
        notes.append("fragility and effective dimensionality skipped: they need two repeats")

    reference_ids = {cid for cid, m in params.items() if m["reference"]}
    diag = {w.name: _diagnostics(w) for w in waves}
    rejected = set()
    for w in waves:
        if diag[w.name] is None:
            notes.append(f"{w.name}: no combo_diagnostics.csv; its specifications are not gated (none dropped as rejected)")
        for r in diag[w.name] or []:
            if (r.get("rejected") or "").strip():
                rejected.add(f"{w.name}/{r['sweep_id']}" if scoped else r["sweep_id"])
    kept_ids = {c for c in params if c not in rejected}
    names = measure_columns(rows1)

    def by_scan(rows, ids):
        d = {}
        for r in rows:
            if r["combo_id"] in ids:
                d.setdefault(r["combo_id"], {})[r["scan"]] = {n: r[n] for n in names}
        return d

    rep1 = by_scan(rows1, kept_ids)

    # leader test
    leader = None
    if len(waves) == 2 and all(diag[w.name] is not None for w in waves):
        def margins(w):
            return {r["sweep_id"]: _num(r.get("discriminability_margin")) for r in diag[w.name]
                    if (r.get("reference") or "").strip().lower() not in ("true", "1")
                    and not (r.get("rejected") or "").strip()}
        discr1 = [v for r in diag[waves[0].name] if (r.get("reference") or "").strip().lower() not in ("true", "1")
                  for v in [_num(r.get("discriminability"))] if v is not None]
        leader = stats.leader_test(margins(waves[0]), margins(waves[1]), discr1)
        notes += leader.pop("notes")
    else:
        notes.append("leader test skipped: it needs exactly two waves with a combo_diagnostics.csv each")

    # fragility and effective dimensionality
    frag = ed = None
    if t2 is not None:
        r2 = by_scan(rows2, kept_ids)
        per = {c: {k: [rep1[c][k], r2[c][k]] for k in rep1[c] if k in r2.get(c, {})} for c in rep1}
        per = {c: s for c, s in per.items() if s}
        frag, ed = stats.fragility(per), stats.effective_dimensionality(per)

    # displacement and reference rho
    displacement = ref_rho = None
    if reference_ids:
        # ponytail: holds every kept specification's repeat-1 edge vectors at once (about 11 GB for 1000
        # specifications x 100 subjects at AAL3 size); stream per wave, keeping only the reference's vectors, if sweeps get that large
        vectors = {}
        for w in waves:
            for combo in sorted((w / "combos").glob("sweep_*")):
                cid = f"{w.name}/{combo.name}" if scoped else combo.name
                if cid in kept_ids:
                    mats = collect_matrices(combo).get((side["atlas"], side["metric"]), {})
                    vectors[cid] = {k: edge_vector(reps[0]) for k, reps in mats.items() if reps}
        displacement = stats.reference_displacement(vectors, reference_ids & kept_ids)
        ref_rho = stats.reference_rho(rep1, reference_ids & kept_ids)
        gone = ", ".join(sorted(reference_ids & rejected))
        if not reference_ids & kept_ids:
            notes.append(f"displacement and reference rho skipped: the reference was rejected by the gate ({gone})")
        else:
            if gone:
                notes.append(f"reference rejected by the gate ({gone}); displacement and reference rho use the remaining reference(s)")
            notes += [s["note"] for s in (displacement, ref_rho) if s.get("note")]
    else:
        notes.append("displacement and reference rho skipped: the sweep has no reference specification")

    # density relation, reference verdict and percentile
    rows_all = rows1 + (rows2 if t2 is not None else [])
    cols = ["density"] + [n for n in names if n != "density"]
    by_combo_rows = {}
    for r in rows_all:
        if r["combo_id"] in kept_ids:
            by_combo_rows.setdefault(r["combo_id"], []).append(r)
    by_sweep, wave_density = {}, {}
    for cid, rs in by_combo_rows.items():
        wave_density.setdefault(stats.wave_of(cid), {})[cid] = float(np.mean([r["density"] for r in rs]))
        if cid not in reference_ids:                       # the relation is over the non-reference specifications
            by_sweep.setdefault(cid.split("/", 1)[-1], []).extend(rs)
    spec_means = {sid: {n: float(np.mean([r[n] for r in rs])) for n in cols} for sid, rs in by_sweep.items()}
    relation = {"relations": stats.density_relation(spec_means, set()),
                "reference_rejected": bool(reference_ids & rejected),
                "reference_density_percentile": stats.reference_percentile(
                    wave_density, {stats.wave_of(c): c for c in reference_ids if c in kept_ids})}

    report = {"sweep": {"optimize_dir": str(optimize_dir), "atlas": side["atlas"], "metric": side["metric"],
                        "waves": [w.name for w in waves], "n_specifications": n_specs,
                        "n_rejected": len(rejected), "reference": sorted(reference_ids)},
              "leader": leader, "fragility": frag, "effective_dimensionality": ed,
              "displacement": displacement, "reference_rho": ref_rho, "density_relation": relation,
              "notes": list(dict.fromkeys(notes))}
    report = _clean(report)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    row = flat_row(report)
    with open(out / "report.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return report


COLUMNS = {
    "leader": ["n_candidates", "discriminability_saturated", "cross_wave_rho", "winner_wave1", "winner_wave2",
               "winners_agree", "top_two_separation", "wave_variation", "winner_determined"],
    "fragility": ["seed_rho", "param_rho", "gap", "n_measures", "n_seed_pairs", "n_param_pairs",
                  "n_seed_measures", "n_param_measures", "min_n_subjects"],
    "effective_dimensionality": ["effective_dimensionality", "n_ed_rows", "n_ed_subjects", "n_ed_dropped"],
    "displacement": ["reference_over_between_subject", "n_reference_pairs"],
    "reference_rho": ["reference_rho", "n_reference_ordering_pairs"],
}


RELATION_COLUMNS = [f"{m}_vs_density" for m in (
    "global_efficiency(binary)", "clustering_coeff_average(binary)", "small_worldness(binary)",
    "clustering_coeff_average(weighted)", "global_efficiency(weighted)", "modularity")]


def flat_row(report: dict) -> dict:
    """One flat row of scalars with fixed column names (None where a statistic was skipped)."""
    row = {}
    for section, cols in COLUMNS.items():
        data = report.get(section) or {}
        row.update({c: data.get(c) for c in cols})
    rel = report.get("density_relation") or {}
    row.update({c: (rel.get("relations") or {}).get(c) for c in RELATION_COLUMNS})
    row["reference_rejected"] = rel.get("reference_rejected")
    row["reference_density_percentile"] = rel.get("reference_density_percentile")
    row["notes"] = " | ".join(report.get("notes") or [])
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opticonn report", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    ap.add_argument("-o", "--output-dir", default=None, help="default <input>/report")
    ap.add_argument("--atlas", default=None)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--recompute", action="store_true", help="rebuild the cached per-subject tables")
    a = ap.parse_args(argv)
    try:
        report = build_report(a.input, a.output_dir, a.atlas, a.metric, a.workers, a.recompute)
    except (ValueError, OSError) as err:
        raise SystemExit(f"opticonn report: {err}")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
