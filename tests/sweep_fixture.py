# tests/sweep_fixture.py
"""A tiny synthetic sweep in the layout OptiConn's collect_matrices reads (no DSI Studio needed)."""

import csv
import json
from pathlib import Path

import numpy as np
import scipy.io

N = 16


def _matrix(rng, n_edges):
    iu = np.triu_indices(N, k=1)
    chosen = rng.permutation(len(iu[0]))[:n_edges]
    w = rng.random(n_edges) + 0.5
    m = np.zeros((N, N))
    m[iu[0][chosen], iu[1][chosen]] = w
    m[iu[1][chosen], iu[0][chosen]] = w
    return m


def make_sweep(root, specs, n_subjects=15, wave="wave1", metrics=("count",), base_edges=40,
               repeats=1, margin=None):
    """specs: [(parameters, reference, edge_offset)]. Subject i has base_edges + offset + 2*i edges
    in every specification and repeat, so density (and the rank of every subject on it) is known
    exactly; the random edges differ between repeats. Edge counts are capped by the 120 possible
    edges at N=16: keep base_edges + offset + 2*(n_subjects - 1) <= 120. An offset of 50 with <= 15
    subjects gives mean density about 0.87, above the 0.6 gate.

    margin: optional callable index -> discriminability margin; if given, a combo_diagnostics.csv is
    written for the wave (specifications with offset >= 50 are marked rejected, margin blank)."""
    diag = []
    for index, (params, reference, offset) in enumerate(specs, start=1):
        combo = Path(root) / wave / "combos" / f"sweep_{index:04d}"
        for rep in range(1, repeats + 1):
            for i in range(n_subjects):
                rng = np.random.default_rng(1000 * index + i + 100000 * (rep - 1))
                d = combo / f"rep_{rep}" / "AAL3"
                d.mkdir(parents=True, exist_ok=True)
                for metric in metrics:
                    scipy.io.savemat(d / f"sub-{i + 1:02d}_ses-1_dwi.qsdr.{metric}..pass.connectivity.mat",
                                     {"connectivity": _matrix(rng, base_edges + offset + 2 * i)})
        (combo / "diagnostics.json").write_text(json.dumps(
            {"status": "ok", "wave": wave, "combo_index": index, "sampler": "grid",
             "reference": reference, "parameters": params}))
        rejected = offset >= 50
        diag.append({"sweep_id": f"sweep_{index:04d}", "reference": str(bool(reference)),
                     "discriminability": 1.0,
                     "discriminability_margin": "" if rejected or margin is None else margin(index),
                     "rejected": "density 0.87 outside [0.02, 0.6]" if rejected else ""})
    if margin is not None:
        with open(Path(root) / wave / "combo_diagnostics.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diag[0]))
            writer.writeheader()
            writer.writerows(diag)
    return Path(root)


def add_session(measures_dir):
    """Append a second scan of the first subject of the first specification to measures.csv."""
    path = Path(measures_dir) / "measures.csv"
    lines = path.read_text().splitlines()
    first = lines[1].split(",")
    first[2] = first[2] + "_again"
    path.write_text("\n".join(lines + [",".join(first)]) + "\n")
