#!/usr/bin/env python3
"""Compute graph/network measures from a connectivity matrix CSV.

This fills the gap for MRtrix-derived connectomes: DSI Studio can emit
`.network_measures.txt`, but MRtrix typically does not.

Outputs a TAB-separated key/value file ending with `.network_measures.csv`
that is compatible with `scripts/aggregate_network_measures.py`.

Metrics emitted (when possible):
- density
- global_efficiency(binary)
- clustering_coeff_average(binary)
- small_worldness(binary)
- global_efficiency(weighted)
- clustering_coeff_average(weighted)

Notes:
- Small-worldness uses NetworkX `sigma()` and can be slow; for pilot runs,
  keep `--smallworld-nrand` small.
- Weighted efficiency uses distances = 1/weight for weight > 0.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
import networkx as nx


def _read_connectivity_csv(path: Path) -> Tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path, index_col=0)
    labels = [str(x) for x in df.index.tolist()]
    mat = df.to_numpy(dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure square
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Connectivity matrix must be square; got {mat.shape}")
    # Zero diagonal
    np.fill_diagonal(mat, 0.0)
    return mat, labels


def _density(mat: np.ndarray) -> float:
    n = mat.shape[0]
    if n <= 1:
        return 0.0
    # undirected density (count nonzero off-diagonal / possible undirected edges)
    upper = np.triu(mat, k=1)
    m = float(np.sum(upper > 0))
    possible = n * (n - 1) / 2
    return float(m / possible) if possible > 0 else 0.0


def _binary_graph(mat: np.ndarray) -> nx.Graph:
    bin_adj = (mat > 0).astype(int)
    np.fill_diagonal(bin_adj, 0)
    G = nx.from_numpy_array(bin_adj)
    return G


def _weighted_graph(mat: np.ndarray) -> nx.Graph:
    # Use undirected graph; weights are edge weights.
    W = mat.copy()
    np.fill_diagonal(W, 0.0)
    G = nx.from_numpy_array(W)
    # NetworkX stores weights under 'weight'
    return G


def _invert_to_strength_from_distance(dist: np.ndarray) -> np.ndarray:
    """Convert a distance-like matrix into a strength-like matrix.

    For dist>0, strength = 1/dist. For dist<=0, strength=0.
    """
    strength = np.zeros_like(dist, dtype=float)
    mask = dist > 0
    strength[mask] = 1.0 / dist[mask]
    np.fill_diagonal(strength, 0.0)
    return strength


def _global_efficiency_weighted_from_strength(mat_strength: np.ndarray) -> float:
    # Efficiency = average over i!=j of 1/d_ij, where d_ij is shortest path
    # length in a distance graph; distance = 1/weight.
    n = mat.shape[0]
    if n <= 1:
        return 0.0

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add weighted edges; store distance for path lengths
    rows, cols = np.where(np.triu(mat_strength, k=1) > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        w = float(mat_strength[i, j])
        if w <= 0:
            continue
        G.add_edge(i, j, weight=w, distance=(1.0 / w))

    # All-pairs shortest paths by distance
    total = 0.0
    count = 0
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="distance"))
    for i in range(n):
        li = lengths.get(i, {})
        for j in range(n):
            if i == j:
                continue
            d = li.get(j)
            if d is None or d <= 0:
                continue
            total += 1.0 / d
            count += 1

    # Normalize by number of ordered pairs
    denom = n * (n - 1)
    if denom <= 0:
        return 0.0

    # If graph disconnected, count < denom. We still divide by denom to keep
    # efficiency comparable and penalize disconnectedness.
    return float(total / denom)


def _global_efficiency_weighted_from_distance(dist: np.ndarray) -> float:
    """Weighted efficiency where matrix entries represent distances (smaller is better)."""
    n = dist.shape[0]
    if n <= 1:
        return 0.0

    G = nx.Graph()
    G.add_nodes_from(range(n))

    rows, cols = np.where(np.triu(dist, k=1) > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        d = float(dist[i, j])
        if d <= 0:
            continue
        # store distance for dijkstra; keep weight as inverse distance for any weighted algorithms
        G.add_edge(i, j, distance=d, weight=(1.0 / d))

    total = 0.0
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="distance"))
    for i in range(n):
        li = lengths.get(i, {})
        for j in range(n):
            if i == j:
                continue
            d = li.get(j)
            if d is None or d <= 0:
                continue
            total += 1.0 / d

    denom = n * (n - 1)
    return float(total / denom) if denom > 0 else 0.0


def _small_world_sigma(G: nx.Graph, nrand: int, seed: int) -> float:
    # sigma() can fail on graphs with no edges or disconnected graphs.
    if G.number_of_nodes() < 3 or G.number_of_edges() == 0:
        return float("nan")
    try:
        return float(nx.algorithms.smallworld.sigma(G, nrand=nrand, niter=1, seed=seed))
    except Exception:
        return float("nan")


def compute_measures(
    connectivity_csv: Path,
    compute_smallworld: bool,
    smallworld_nrand: int,
    seed: int,
    weight_type: Literal["strength", "distance"] = "strength",
) -> Dict[str, float]:
    mat, _labels = _read_connectivity_csv(connectivity_csv)

    measures: Dict[str, float] = {}
    measures["density"] = _density(mat)

    # Binary
    Gbin = _binary_graph(mat)
    measures["global_efficiency(binary)"] = float(nx.global_efficiency(Gbin))
    measures["clustering_coeff_average(binary)"] = float(nx.average_clustering(Gbin))
    if compute_smallworld:
        measures["small_worldness(binary)"] = _small_world_sigma(
            Gbin, nrand=smallworld_nrand, seed=seed
        )

    # Weighted
    if weight_type == "strength":
        Gwt = _weighted_graph(mat)
        try:
            measures["clustering_coeff_average(weighted)"] = float(
                nx.average_clustering(Gwt, weight="weight")
            )
        except Exception:
            measures["clustering_coeff_average(weighted)"] = float("nan")

        try:
            measures["global_efficiency(weighted)"] = _global_efficiency_weighted_from_strength(
                mat
            )
        except Exception:
            measures["global_efficiency(weighted)"] = float("nan")

    elif weight_type == "distance":
        # Treat matrix entries as distances; convert to strengths for clustering only.
        strength = _invert_to_strength_from_distance(mat)
        Gwt = _weighted_graph(strength)
        try:
            measures["clustering_coeff_average(weighted)"] = float(
                nx.average_clustering(Gwt, weight="weight")
            )
        except Exception:
            measures["clustering_coeff_average(weighted)"] = float("nan")

        try:
            measures["global_efficiency(weighted)"] = _global_efficiency_weighted_from_distance(
                mat
            )
        except Exception:
            measures["global_efficiency(weighted)"] = float("nan")

    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    return measures


def write_network_measures_csv(measures: Dict[str, float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for key in sorted(measures.keys()):
            val = measures[key]
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            f.write(f"{key}\t{val}\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute network measures from a connectivity matrix CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("connectivity_csv", type=Path, help="Input *.connectivity.csv")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output *.network_measures.csv (default: alongside input)",
    )
    p.add_argument(
        "--smallworld",
        action="store_true",
        help="Compute small-worldness(binary) via NetworkX sigma()",
    )
    p.add_argument(
        "--smallworld-nrand",
        type=int,
        default=20,
        help="Number of random graphs for sigma()",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--weight-type",
        choices=["strength", "distance"],
        default="strength",
        help="Interpretation of matrix values for weighted measures",
    )

    args = p.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = args.connectivity_csv.with_suffix("")
        out_path = Path(str(out_path) + ".network_measures.csv")

    measures = compute_measures(
        args.connectivity_csv,
        compute_smallworld=args.smallworld,
        smallworld_nrand=args.smallworld_nrand,
        seed=args.seed,
        weight_type=args.weight_type,
    )
    write_network_measures_csv(measures, out_path)

    print(f"Wrote network measures: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
