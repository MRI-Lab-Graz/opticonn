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
