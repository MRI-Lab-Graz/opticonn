# scripts/measures_table.py
"""Tidy table of graph measures per subject and specification, from a completed sweep.

    opticonn measures -i <optimize> [--atlas A] [--metric M] [--repeat N] [--extended]

Writes <optimize>/measures/measures.csv, params.csv and measures.json; `outcome` and `pca` read
them, so the slow step (reading every matrix) runs once.
Design: docs/superpowers/specs/2026-10-22-outcome-sensitivity-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np

from scripts.compute_network_measures_from_connectivity import _binary_graph, measures_from_matrix
from scripts.reliability import collect_matrices
from scripts.utils.discovery import parse_subject_session

EXTRA = ["local_efficiency(binary)", "mean_strength(weighted)", "characteristic_path_length(binary)",
         "mean_betweenness(binary)", "degree_assortativity(binary)"]
ID_COLUMNS = ("combo_id", "subject", "scan")


def extra_measures(mat: np.ndarray) -> dict:
    """The five extra global measures of one connectivity matrix (non-finite -> 0, zero diagonal)."""
    mat = np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).copy()
    np.fill_diagonal(mat, 0.0)
    g = _binary_graph(mat)
    big = g.subgraph(max(nx.connected_components(g), key=len)) if g.number_of_edges() else g
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # constant degrees: assortativity is NaN
        return {
            "local_efficiency(binary)": float(nx.local_efficiency(g)),
            "mean_strength(weighted)": float(mat.sum(axis=1).mean()),
            "characteristic_path_length(binary)": (float(nx.average_shortest_path_length(big))
                                                   if big.number_of_nodes() > 1 else float("nan")),
            "mean_betweenness(binary)": float(np.mean(list(nx.betweenness_centrality(g).values()))),
            "degree_assortativity(binary)": float(nx.degree_assortativity_coefficient(g)),
        }


def _isolated_fraction(mat: np.ndarray) -> float:
    edges = np.nan_to_num(np.asarray(mat, dtype=float)) != 0
    np.fill_diagonal(edges, False)
    return float(np.mean(~edges.any(axis=0) & ~edges.any(axis=1)))


def matrix_row(mat: np.ndarray, extended: bool) -> dict:
    """The seven standard measures, the isolated-node fraction and, if extended, the five more."""
    row = measures_from_matrix(mat)
    row["isolated_fraction"] = _isolated_fraction(mat)
    if extended:
        row.update(extra_measures(mat))
    return row


def find_wave(optimize_dir, wave: str | None = None) -> Path:
    waves = sorted({p.parent for p in Path(optimize_dir).glob("*/combos")})
    if wave:
        match = [w for w in waves if w.name == wave]
        if not match:
            raise ValueError(f"no wave named {wave}; waves are {', '.join(w.name for w in waves)}")
        return match[0]
    if len(waves) != 1:
        raise ValueError(f"{optimize_dir} has {len(waves)} waves ({', '.join(w.name for w in waves)}); "
                         "choose one with --wave")
    return waves[0]


def find_waves(optimize_dir, wave: str | None = None) -> list[Path]:
    """The wave directories a command works on: every wave for `all`, otherwise the one `find_wave` picks."""
    if wave == "all":
        waves = sorted({p.parent for p in Path(optimize_dir).glob("*/combos")})
        if not waves:
            raise ValueError(f"{optimize_dir}: no waves found")
        return waves
    return [find_wave(optimize_dir, wave)]


def read_meta(wave_dir: Path) -> dict:
    """{combo name: {"reference": bool, "parameters": dict}} from each combo's diagnostics.json."""
    meta = {}
    for path in sorted(Path(wave_dir).glob("combos/sweep_*/diagnostics.json")):
        d = json.loads(path.read_text())
        meta[path.parent.name] = {"reference": bool(d.get("reference")), "parameters": d.get("parameters") or {}}
    return meta


def _combo(args):
    name, combo_dir, atlas, metric, repeat, extended = args
    rows = []
    for scan, reps in collect_matrices(combo_dir).get((atlas, metric), {}).items():
        if len(reps) < repeat:
            continue
        rows.append({"combo_id": name, "subject": parse_subject_session(scan)[0] or scan, "scan": scan,
                     **matrix_row(reps[repeat - 1], extended)})
    return rows


def read_meta_waves(waves: list[Path]) -> dict:
    """read_meta over several waves; ids are wave-scoped (`wave1/sweep_0001`) when there is more than one."""
    scoped = len(waves) > 1
    return {(f"{w.name}/{cid}" if scoped else cid): m for w in waves for cid, m in read_meta(w).items()}


def write_measures(optimize_dir, out_dir=None, atlas=None, metric=None, repeat=1, extended=False,
                   workers=8, wave=None) -> Path:
    if repeat < 1:
        raise ValueError("--repeat must be 1 or more")
    optimize_dir = Path(optimize_dir)
    waves = find_waves(optimize_dir, wave)
    scoped = len(waves) > 1
    combos = [(f"{w.name}/{c.name}" if scoped else c.name, c)
              for w in waves for c in sorted((w / "combos").glob("sweep_*"))]
    if not combos:
        raise ValueError(f"{optimize_dir}: no specifications found")
    pairs = next((sorted(m) for m in map(collect_matrices, (c for _, c in combos)) if m), None)
    if pairs is None:
        raise ValueError("no specification has connectivity matrices")
    chosen = [p for p in pairs if (atlas in (None, p[0])) and (metric in (None, p[1]))]
    if len(chosen) != 1:
        found = ", ".join(f"{a}/{m}" for a, m in pairs)
        raise ValueError(f"more than one atlas/metric in the sweep ({found}); choose with --atlas/--metric"
                         if chosen else f"no atlas/metric matches --atlas/--metric; the sweep has {found}")
    atlas, metric = chosen[0]
    jobs = [(name, c, atlas, metric, repeat, extended) for name, c in combos]
    if workers <= 1:                                        # serial: no fork, no pool overhead
        parts = map(_combo, jobs)
    else:
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_combo, jobs))
    parts = list(parts)
    rows = [r for part in parts for r in part]
    missing = sum(not part for part in parts)
    if not rows:
        raise ValueError(f"no subject has repeat {repeat}")
    out = Path(out_dir) if out_dir else optimize_dir / "measures"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "measures.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = read_meta_waves(waves)
    names = list(dict.fromkeys(k for m in meta.values() for k in m["parameters"]))
    with open(out / "params.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["combo_id", "reference", *names])
        writer.writeheader()
        for cid, m in meta.items():
            writer.writerow({"combo_id": cid, "reference": m["reference"],
                             **{k: "" if v is None else v for k, v in m["parameters"].items()}})
    try:
        from importlib.metadata import version
        ver = version("opticonn")
    except Exception:
        ver = "unknown"
    (out / "measures.json").write_text(json.dumps(
        {"atlas": atlas, "metric": metric, "repeat": repeat, "extended": extended,
         "n_specifications": len(combos), "n_specifications_without_matrices": missing, "wave": waves[0].name if len(waves) == 1 else "all", "waves": [w.name for w in waves],
         "wave_setting": wave, "opticonn_version": ver}, indent=2) + "\n")
    return out


def read_measures(measures_dir) -> list[dict]:
    path = Path(measures_dir) / "measures.csv"
    if not path.exists():
        raise ValueError(f"{path} not found; run `opticonn measures -i <optimize>` first")
    with open(path, newline="") as handle:
        return [{k: (v if k in ID_COLUMNS else float(v)) for k, v in r.items()} for r in csv.DictReader(handle)]


def check_one_scan_per_subject(rows: list[dict]) -> None:
    seen: dict = {}
    for r in rows:
        first = seen.setdefault((r["combo_id"], r["subject"]), r["scan"])
        if first != r["scan"]:
            raise ValueError(f"{r['subject']} has two scans in {r['combo_id']} ({first}, {r['scan']}); "
                             "outcome and pca need one scan per subject; measures.csv has several scans for it (sessions?)")


def measure_columns(rows: list[dict]) -> list[str]:
    return [k for k in rows[0] if k not in ID_COLUMNS and k != "isolated_fraction"]


def read_params(measures_dir) -> dict:
    """{combo: {"reference": bool, "parameters": {name: str}}} from params.csv."""
    with open(Path(measures_dir) / "params.csv", newline="") as handle:
        return {r["combo_id"]: {"reference": r["reference"] == "True",
                                "parameters": {k: v for k, v in r.items() if k not in ("combo_id", "reference")}}
                for r in csv.DictReader(handle)}


def tables_current(out_dir, atlas, metric, repeat, wave, n_specifications) -> bool:
    """True if out_dir holds measure tables written with these settings for a sweep of this size."""
    out = Path(out_dir)
    if not ((out / "measures.csv").exists() and (out / "measures.json").exists()):
        return False
    s = json.loads((out / "measures.json").read_text())
    return (atlas in (None, s["atlas"]) and metric in (None, s["metric"]) and s["repeat"] == repeat
            and s.get("wave_setting") == wave and s["n_specifications"] == n_specifications)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opticonn measures", description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="a sweep's optimize/ directory")
    ap.add_argument("-o", "--output-dir", default=None, help="default <input>/measures")
    ap.add_argument("--wave", default=None, help="a wave name, or 'all' for every wave (ids are then wave-scoped)")
    ap.add_argument("--atlas", default=None)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--extended", action="store_true", help="add local efficiency, mean strength, path length, "
                    "mean betweenness and degree assortativity")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    try:
        out = write_measures(a.input, a.output_dir, a.atlas, a.metric, a.repeat, a.extended, a.workers, a.wave)
    except (ValueError, OSError) as err:
        raise SystemExit(f"opticonn measures: {err}")
    side = json.loads((out / "measures.json").read_text())
    if side["n_specifications_without_matrices"]:
        print(f"{side['n_specifications_without_matrices']} of {side['n_specifications']} specifications had no "
              "matrices for atlas/metric and are absent from the table")
    print(f"wrote {out}/measures.csv, params.csv, measures.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
