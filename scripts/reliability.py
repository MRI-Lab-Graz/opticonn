"""Reliability-based scoring for tractography parameter sweeps.

A parameter set is preferred when connectomes from repeated tracking runs of
the same subject agree more than connectomes of different subjects
(discriminability), after rejecting implausible graphs (density and
isolated-node gates). Ties are broken by run-to-run repeatability.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.io

DEFAULT_GATES = {"density_range": [0.02, 0.6], "max_isolated_fraction": 0.1}

_MATRIX_NAME = re.compile(r"\.([^.]+)\.\.(?:pass|end)\.connectivity\.mat$")

# Known ceiling: only these three metrics have known r2r keys in newer DSI Studio's combined
# .connectivity.mat output; add an entry here when a sweep config uses another
# connectivity_value (e.g. ncount2) and the combined-format tests start missing it.
_COMBINED_METRIC_KEYS = {
    "count": "number of tracts r2r",
    "fa": "dti_fa r2r",
    "qa": "qa r2r",
}


def load_matrix(path: Path) -> np.ndarray:
    return np.asarray(scipy.io.loadmat(str(path))["connectivity"], dtype=float)


def _load_combined(path: Path) -> dict[str, np.ndarray]:
    """Newer DSI Studio writes one <atlas>.connectivity.mat per subject/rep with
    per-metric data under fixed r2r keys instead of separate per-metric files."""
    data = scipy.io.loadmat(str(path))
    return {
        metric: np.asarray(data[key], dtype=float)
        for metric, key in _COMBINED_METRIC_KEYS.items()
        if key in data
    }


def edge_vector(matrix: np.ndarray) -> np.ndarray:
    """Upper-triangle edge weights, log-compressed so a few huge counts don't dominate."""
    return np.log1p(np.abs(matrix[np.triu_indices(matrix.shape[0], k=1)]))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 1.0
    return 1.0 - float(np.corrcoef(a, b)[0, 1])


def discriminability(mats: dict[str, list[np.ndarray]]) -> float:
    """P(a repeat of the same subject is closer than a repeat of another subject).

    0.5 = chance, 1.0 = every subject identifiable above tracking noise.
    NaN when fewer than two subjects, or no subject has two repeats.
    """
    # Known ceiling: O(subjects² x repeats²) correlations; fine for 3-10 sweep subjects, vectorise past ~50.
    vecs = {s: [edge_vector(m) for m in ms] for s, ms in mats.items()}
    wins, total = 0.0, 0
    for subject, repeats in vecs.items():
        others = [v for other, vs in vecs.items() if other != subject for v in vs]
        for i, a in enumerate(repeats):
            between = [_distance(a, c) for c in others]
            for j, b in enumerate(repeats):
                if i == j:
                    continue
                within = _distance(a, b)
                wins += sum(1.0 if within < d else 0.5 if within == d else 0.0 for d in between)
                total += len(between)
    return wins / total if total else float("nan")


def repeatability(mats: dict[str, list[np.ndarray]]) -> float:
    """Mean correlation between repeated runs of the same subject (1.0 = no tracking noise)."""
    corrs = [
        1.0 - _distance(edge_vector(ms[i]), edge_vector(ms[j]))
        for ms in mats.values()
        for i in range(len(ms))
        for j in range(i + 1, len(ms))
    ]
    return float(np.mean(corrs)) if corrs else float("nan")


def graph_stats(mats: dict[str, list[np.ndarray]]) -> tuple[float, float]:
    """Mean edge density and mean fraction of nodes without any edge."""
    densities, isolated = [], []
    for m in (m for ms in mats.values() for m in ms):
        edges = m != 0
        np.fill_diagonal(edges, False)
        n = m.shape[0]
        densities.append(np.triu(edges, 1).sum() / (n * (n - 1) / 2))
        isolated.append(float(np.mean(~edges.any(axis=0) & ~edges.any(axis=1))))
    return float(np.mean(densities)), float(np.mean(isolated))


def gate_reason(density: float, isolated: float, density_range, max_isolated_fraction: float) -> str:
    """Empty string if the graph is plausible, otherwise why it was rejected."""
    lo, hi = density_range
    if not lo <= density <= hi:
        return f"density {density:.3f} outside [{lo}, {hi}]"
    if isolated > max_isolated_fraction:
        return f"isolated nodes {isolated:.3f} > {max_isolated_fraction}"
    return ""


def collect_matrices(combo_dir: Path) -> dict[tuple[str, str], dict[str, list[np.ndarray]]]:
    """{(atlas, metric): {subject: [matrix per repeat]}} from <combo_dir>/rep_*/**/<atlas>/*.connectivity.mat.

    Reads two DSI Studio output layouts:
    - legacy: one file per metric, name matches `_MATRIX_NAME`, matrix under the "connectivity" key.
    - newer (combined): one file per subject/rep with no metric segment in the name, holding
      several metrics under fixed r2r keys (see `_COMBINED_METRIC_KEYS`).
    """
    found: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for rep_dir in sorted(Path(combo_dir).glob("rep_*")):
        for path in sorted(rep_dir.rglob("*.connectivity.mat")):
            atlas = path.parent.name
            subject = path.name.split(f"_{atlas}.")[0].split(".")[0]
            match = _MATRIX_NAME.search(path.name)
            metrics = {match.group(1): load_matrix(path)} if match else _load_combined(path)
            for metric, matrix in metrics.items():
                found.setdefault((atlas, metric), {}).setdefault(subject, []).append(matrix)
    return found


def score_combo(combo_dir: Path, reliability_cfg: dict) -> list[dict]:
    """One row per (atlas, metric) found under combo_dir."""
    gates = {**DEFAULT_GATES, **(reliability_cfg or {})}
    rows = []
    for (atlas, metric), mats in sorted(collect_matrices(combo_dir).items()):
        density, isolated = graph_stats(mats)
        rows.append(
            {
                "atlas": atlas,
                "connectivity_metric": metric,
                "n_subjects": len(mats),
                "n_repeats": min(len(ms) for ms in mats.values()),
                "discriminability": discriminability(mats),
                "repeatability": repeatability(mats),
                "density": density,
                "isolated_fraction": isolated,
                "rejected": gate_reason(density, isolated, gates["density_range"], gates["max_isolated_fraction"]),
            }
        )
    return rows


def rank(rows: list[dict]) -> list[dict]:
    """Plausible candidates, best first: discriminability, then repeatability, then fewer tracts."""
    usable = [r for r in rows if not r["rejected"] and not np.isnan(r["discriminability"])]
    return sorted(usable, key=lambda r: (-r["discriminability"], -r["repeatability"], r.get("tract_count") or 0))


def loo_top1_frequency(candidates: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, float]:
    """Share of leave-one-subject-out rankings in which each candidate is first.

    NaN for all candidates when fewer than 3 subjects are shared.
    """
    shared = sorted(set.intersection(*(set(m) for m in candidates.values()))) if candidates else []
    if len(shared) < 3:
        return dict.fromkeys(candidates, float("nan"))
    wins = dict.fromkeys(candidates, 0)
    for left_out in shared:
        scores = {
            key: discriminability({s: v for s, v in mats.items() if s in shared and s != left_out})
            for key, mats in candidates.items()
        }
        wins[max(scores, key=lambda k: np.nan_to_num(scores[k], nan=-1.0))] += 1
    return {key: n / len(shared) for key, n in wins.items()}
