"""Reliability-based scoring for tractography parameter sweeps.

A parameter set is preferred when connectomes from repeated tracking runs of
the same subject agree more than connectomes of different subjects
(discriminability), after rejecting implausible graphs (density and
isolated-node gates). Ties are broken by run-to-run repeatability.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

DEFAULT_GATES = {"density_range": [0.02, 0.6], "max_isolated_fraction": 0.1}

_MATRIX_NAME = re.compile(r"\.([^.]+)\.\.(?:pass|end)\.connectivity\.mat$")

# MRtrix3 backend layout (scripts/mrtrix_tune.py): one metric per
# `<subject>_<atlas>.<metric>.connectivity.csv`, written by
# scripts.utils.mrtrix.write_opticonn_connectivity_csv -- a labelled CSV, not a .mat.
_CSV_MATRIX_NAME = re.compile(r"\.([^.]+)\.connectivity\.csv$")

# Known ceiling: only these three metrics have known r2r keys in newer DSI Studio's
# combined .connectivity.mat output; add an entry here when a sweep config uses another
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
    # Known ceiling: O(subjects² × repeats²) correlations; fine for 3-10 sweep
    # subjects, vectorise if this is ever run past ~50.
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


def _load_csv_matrix(path: Path) -> np.ndarray:
    """MRtrix3 backend output: `scripts.utils.mrtrix.write_opticonn_connectivity_csv`
    writes one metric per CSV, labelled rows/cols with the region index/name as the
    first column -- read it back as a plain numeric matrix."""
    return pd.read_csv(path, index_col=0).to_numpy(dtype=float)


def collect_matrices(combo_dir: Path) -> dict[tuple[str, str], dict[str, list[np.ndarray]]]:
    """{(atlas, metric): {subject: [matrix per repeat]}} from <combo_dir>/rep_*/**/<atlas>/*.connectivity.{mat,csv}.

    Reads three output layouts:
    - legacy DSI Studio: one .mat file per metric, name matches `_MATRIX_NAME`, matrix
      under the "connectivity" key.
    - newer (combined) DSI Studio: one .mat file per subject/rep with no metric segment
      in the name, holding several metrics under fixed r2r keys (see `_COMBINED_METRIC_KEYS`).
    - MRtrix3 backend: one .csv file per metric (see `_load_csv_matrix`); there is no
      combined-CSV equivalent, so a filename that doesn't match `_CSV_MATRIX_NAME` is skipped.

    Returns one matrix per (atlas, metric, scan) per repeat. DSI Studio writes each metric
    twice per scan -- inside the combined .connectivity.mat and again as a converted
    per-metric .csv -- and loading both duplicated every repeat count. The .mat is the
    source of truth; a .csv only supplies metrics no .mat provided (MRtrix3 is CSV-only).
    """
    found: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}
    for rep_dir in sorted(Path(combo_dir).glob("rep_*")):
        # One matrix per (atlas, metric, scan) per repeat. DSI Studio writes each
        # metric twice per scan -- inside the combined <atlas>.connectivity.mat and
        # again as a converted per-metric .connectivity.csv -- and loading both
        # made every repeat count twice (n_repeats 4 instead of 2, repeatability
        # inflated, and a scan with one surviving repeat still passing the >=2
        # repeats gate). The .mat is the source of truth so it wins; a .csv only
        # fills metrics no .mat supplied (the MRtrix3 backend writes CSV only).
        supplied: set[tuple[str, str, str]] = set()
        paths = sorted(rep_dir.rglob("*.connectivity.mat")) + sorted(rep_dir.rglob("*.connectivity.csv"))
        for path in paths:
            atlas = path.parent.name
            subject = path.name.split(f"_{atlas}.")[0].split(".")[0]
            if path.suffix == ".csv":
                match = _CSV_MATRIX_NAME.search(path.name)
                if not match:
                    continue
                metrics = {match.group(1): _load_csv_matrix(path)}
            else:
                match = _MATRIX_NAME.search(path.name)
                metrics = {match.group(1): load_matrix(path)} if match else _load_combined(path)
            for metric, matrix in metrics.items():
                if (atlas, metric, subject) in supplied:
                    continue
                supplied.add((atlas, metric, subject))
                found.setdefault((atlas, metric), {}).setdefault(subject, []).append(matrix)
    return found


def score_combo(combo_dir: Path, reliability_cfg: dict) -> list[dict]:
    """One row per (atlas, metric) found under combo_dir."""
    gates = {**DEFAULT_GATES, **(reliability_cfg or {})}
    rows = []
    for (atlas, metric), mats in sorted(collect_matrices(combo_dir).items()):
        density, isolated = graph_stats(mats)
        rejected = gate_reason(density, isolated, gates["density_range"], gates["max_isolated_fraction"])
        repeat_counts = [len(ms) for ms in mats.values()]
        if not rejected:
            if min(repeat_counts) < 2:
                rejected = "fewer than 2 repeats for some subject"
            elif len(set(repeat_counts)) > 1:
                rejected = "unequal repeats across subjects"
        rows.append(
            {
                "atlas": atlas,
                "connectivity_metric": metric,
                "n_subjects": len(mats),
                "n_repeats": min(repeat_counts),
                "discriminability": discriminability(mats),
                "repeatability": repeatability(mats),
                "density": density,
                "isolated_fraction": isolated,
                "rejected": rejected,
            }
        )
    return rows


def rank(rows: list[dict]) -> list[dict]:
    """Plausible candidates, best first: discriminability, then repeatability, then fewer tracts."""
    usable = [r for r in rows if not r["rejected"] and not np.isnan(r["discriminability"])]
    return sorted(usable, key=lambda r: (-r["discriminability"], -r["repeatability"], r.get("tract_count") or 0))


def _desc(value) -> float:
    """Sort key for "bigger is better", with NaN/missing sorted last."""
    if isinstance(value, (int, float)) and not math.isnan(value):
        return -float(value)
    return float("inf")


def rank_with_fallback(rows: list[dict]) -> dict | None:
    """Best row by `rank()`, degrading gracefully when discriminability is NaN.

    `rank()` drops every row with NaN discriminability -- which is every row when
    the pool holds a single subject, even though the reliability gates themselves
    passed. That is not a gate failure, so fall back to the best gate-passing row
    by repeatability, then `quality_score_raw`. Returns None only when every row
    was genuinely gate-rejected (or there were no rows at all).

    Shared by both backends' combo/theta assembly and selection so the DSI and
    MRtrix paths cannot drift apart again.
    """
    ranked = rank(rows)
    if ranked:
        return ranked[0]
    passable = [r for r in rows if not r.get("rejected")]
    if not passable:
        return None
    return sorted(passable, key=lambda r: (_desc(r.get("repeatability")), _desc(r.get("quality_score_raw"))))[0]


def resolve_repeats(reliability_cfg: dict | None, override=None) -> int:
    """Repeat count for a sweep, warning when it is too low to score reliability.

    `score_combo` rejects any candidate with fewer than 2 repeats per subject, so
    `repeats: 1` silently makes every candidate ungradeable -- warn at config-load
    time instead of after hours of compute.
    """
    value = override if override is not None else (reliability_cfg or {}).get("repeats", 2)
    repeats = max(1, int(value))
    if repeats < 2:
        logging.warning(
            "reliability.repeats=%d: discriminability/repeatability need at least "
            "2 repeats per subject; every candidate will be rejected as 'fewer than "
            "2 repeats for some subject'. Set reliability.repeats >= 2.",
            repeats,
        )
    return repeats


def loo_top1_frequency(
    candidates: dict[str, dict[str, list[np.ndarray]]],
    tract_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    """Share of leave-one-subject-out rankings in which each candidate is first.

    Ties on the `rank()` ordering (discriminability, then repeatability, then
    fewer tracts) split the win fractionally between every tied candidate.
    NaN for all candidates when fewer than 3 subjects are shared.
    """
    shared = sorted(set.intersection(*(set(m) for m in candidates.values()))) if candidates else []
    if len(shared) < 3:
        return dict.fromkeys(candidates, float("nan"))
    tract_counts = tract_counts or {}
    wins = dict.fromkeys(candidates, 0.0)
    for left_out in shared:
        held_in = lambda mats: {s: v for s, v in mats.items() if s in shared and s != left_out}
        keys = {
            key: (
                -np.nan_to_num(discriminability(held_in(mats)), nan=-1.0),
                -repeatability(held_in(mats)),
                tract_counts.get(key) or 0,
            )
            for key, mats in candidates.items()
        }
        best = min(keys.values())
        winners = [key for key, k in keys.items() if k == best]
        for key in winners:
            wins[key] += 1.0 / len(winners)
    return {key: n / len(shared) for key, n in wins.items()}
