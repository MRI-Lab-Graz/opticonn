# tests/sweep_fixture.py
"""A tiny synthetic sweep in the layout OptiConn's collect_matrices reads (no DSI Studio needed)."""

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


def make_sweep(root, specs, n_subjects=15, wave="wave1", metrics=("count",), base_edges=40):
    """specs: [(parameters, reference, edge_offset)]. Subject i has base_edges + offset + 2*i edges
    in every specification, so density (and the rank of every subject on it) is known exactly.
    Edge counts are capped by the 120 possible edges at N=16: keep base_edges + offset +
    2*(n_subjects - 1) <= 120. An offset of 50 with <= 15 subjects gives mean density about 0.87,
    above the 0.6 gate."""
    for index, (params, reference, offset) in enumerate(specs, start=1):
        combo = Path(root) / wave / "combos" / f"sweep_{index:04d}"
        for i in range(n_subjects):
            rng = np.random.default_rng(1000 * index + i)
            d = combo / "rep_1" / "AAL3"
            d.mkdir(parents=True, exist_ok=True)
            for metric in metrics:
                scipy.io.savemat(d / f"sub-{i + 1:02d}_ses-1_dwi.qsdr.{metric}..pass.connectivity.mat",
                                 {"connectivity": _matrix(rng, base_edges + offset + 2 * i)})
        (combo / "diagnostics.json").write_text(json.dumps(
            {"status": "ok", "wave": wave, "combo_index": index, "sampler": "grid",
             "reference": reference, "parameters": params}))
    return Path(root)


def add_session(measures_dir):
    """Append a second scan of the first subject of the first specification to measures.csv."""
    path = Path(measures_dir) / "measures.csv"
    lines = path.read_text().splitlines()
    first = lines[1].split(",")
    first[2] = first[2] + "_again"
    path.write_text("\n".join(lines + [",".join(first)]) + "\n")
