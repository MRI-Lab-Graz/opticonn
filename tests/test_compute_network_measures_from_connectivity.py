"""Tests for scripts/compute_network_measures_from_connectivity.py.

Ported from origin/mrtrix-backend-draft (Task 5's MRtrix3 backend needs it to
fill the network-measures gap MRtrix leaves, unlike DSI Studio).
"""

import math

import numpy as np
import pandas as pd
import pytest

from scripts.compute_network_measures_from_connectivity import (
    compute_measures,
    measures_from_matrix,
    write_network_measures_csv,
)


def _write_connectivity_csv(path, matrix, labels=None):
    n = matrix.shape[0]
    labels = labels or [str(i) for i in range(n)]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)


def test_compute_measures_strength_weight_type_does_not_crash(tmp_path):
    # Regression test: the draft's `_global_efficiency_weighted_from_strength`
    # referenced a module-level `mat` name instead of its own `mat_strength`
    # parameter, which raised NameError for every weight_type="strength" call
    # (the default, and what every non-"meanlength" MRtrix metric uses).
    rng = np.random.default_rng(0)
    upper = np.triu(rng.random((6, 6)), 1)
    matrix = upper + upper.T
    csv_path = tmp_path / "sub-1_atlas.count.connectivity.csv"
    _write_connectivity_csv(csv_path, matrix)

    measures = compute_measures(
        csv_path, compute_smallworld=False, smallworld_nrand=5, seed=0, weight_type="strength"
    )

    assert "global_efficiency(weighted)" in measures
    assert not math.isnan(measures["global_efficiency(weighted)"])


def test_compute_measures_density_for_fully_connected_graph(tmp_path):
    matrix = np.ones((4, 4))
    np.fill_diagonal(matrix, 0.0)
    csv_path = tmp_path / "sub-1_atlas.count.connectivity.csv"
    _write_connectivity_csv(csv_path, matrix)

    measures = compute_measures(
        csv_path, compute_smallworld=False, smallworld_nrand=5, seed=0, weight_type="strength"
    )

    assert measures["density"] == pytest.approx(1.0)


def test_write_network_measures_csv_skips_nan_and_inf(tmp_path):
    out_path = tmp_path / "out.network_measures.csv"
    write_network_measures_csv(
        {"density": 0.5, "bad_nan": float("nan"), "bad_inf": float("inf")}, out_path
    )

    text = out_path.read_text()
    assert "density\t0.5" in text
    assert "bad_nan" not in text
    assert "bad_inf" not in text


def _two_communities(n=12):
    """Two dense blocks joined by one weak edge: modularity must be clearly positive."""
    m = np.zeros((n, n))
    half = n // 2
    m[:half, :half] = 5.0
    m[half:, half:] = 5.0
    m[0, half] = m[half, 0] = 0.1
    np.fill_diagonal(m, 0.0)
    return m


def test_measures_from_matrix_matches_compute_measures_via_csv(tmp_path):
    rng = np.random.default_rng(1)
    upper = np.triu(rng.random((10, 10)) * (rng.random((10, 10)) < 0.5), 1)
    matrix = upper + upper.T
    csv_path = tmp_path / "sub-1_atlas.count.connectivity.csv"
    _write_connectivity_csv(csv_path, matrix)

    from_csv = compute_measures(csv_path, compute_smallworld=False, smallworld_nrand=5, seed=0)
    direct = measures_from_matrix(matrix, seed=0)

    assert from_csv.keys() == direct.keys()
    for key in direct:
        assert direct[key] == pytest.approx(from_csv[key], nan_ok=True), key


def test_measures_from_matrix_reports_the_default_measure_set():
    assert set(measures_from_matrix(_two_communities())) == {
        "density",
        "global_efficiency(binary)",
        "clustering_coeff_average(binary)",
        "small_worldness(binary)",
        "clustering_coeff_average(weighted)",
        "global_efficiency(weighted)",
        "modularity",
    }


def test_modularity_is_positive_for_two_communities_and_seed_reproducible():
    first = measures_from_matrix(_two_communities(), seed=7)["modularity"]
    second = measures_from_matrix(_two_communities(), seed=7)["modularity"]
    assert first > 0.3
    assert first == second


def test_modularity_is_nan_not_an_error_for_an_empty_graph():
    assert math.isnan(measures_from_matrix(np.zeros((5, 5)))["modularity"])
