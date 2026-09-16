"""Regression tests for DSI Studio .mat output compatibility.

Newer DSI Studio builds no longer emit a per-metric .mat file named
``<base>_<atlas>.<metric>..pass.connectivity.mat`` with a top-level
``connectivity``/``matrix``/``data`` key. Instead they emit one combined
``<base>_<atlas>.tt.gz.<atlas>.connectivity.mat`` per atlas holding every
metric as ``"<metric> r2r"`` (region-to-region, NxN) and ``"<metric> t2r"``
(tract-to-region, Nx1) arrays. The old parser picked the first available key
regardless of shape, which crashed on the Nx1 arrays and marked every atlas
as failed. See the walkthrough that discovered this for the real DSI Studio
log output.
"""

import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from pathlib import Path

from scripts.extract_connectivity_matrices import ConnectivityExtractor


def _make_extractor(connectivity_values):
    return ConnectivityExtractor(config={"connectivity_values": connectivity_values})


def test_convert_mat_to_csv_legacy_connectivity_key(tmp_path):
    """Old DSI Studio .mat files with a top-level 'connectivity' key still work."""
    matrix = np.arange(9, dtype=float).reshape(3, 3)
    mat_path = tmp_path / "sub01_Atlas.fa..pass.connectivity.mat"
    scipy_io.savemat(str(mat_path), {"connectivity": matrix})

    extractor = _make_extractor(["fa"])
    result = extractor.convert_mat_to_csv(mat_path, "Atlas")

    assert result["success"] is True
    assert result["matrix_shape"] == (3, 3)


def test_convert_mat_to_csv_new_bundled_r2r_format(tmp_path):
    """New DSI Studio .mat bundles every metric as '<metric> r2r' (NxN) plus
    a '<metric> t2r' (Nx1) vector; requesting connectivity_value 'fa' must
    resolve to the 'dti_fa r2r' square matrix, not crash on the Nx1 vector."""
    n = 4
    r2r = np.arange(n * n, dtype=float).reshape(n, n)
    t2r = np.arange(n, dtype=float).reshape(n, 1)
    mat_path = tmp_path / "sub01_Atlas.tt.gz.Atlas.connectivity.mat"
    scipy_io.savemat(
        str(mat_path),
        {
            "dti_fa t2r": t2r,
            "dti_fa r2r": r2r,
            "qa t2r": t2r,
            "qa r2r": r2r + 1,
            "name": np.array(["region1;region2"]),
        },
    )

    extractor = _make_extractor(["fa"])
    result = extractor.convert_mat_to_csv(mat_path, "Atlas")

    assert result["success"] is True
    assert result["matrix_shape"] == (n, n)
    saved = np.loadtxt(result["simple_csv_path"], delimiter=",")
    np.testing.assert_array_equal(saved, r2r)


def test_convert_mat_to_csv_new_format_unknown_metric_falls_back_to_square_matrix(tmp_path):
    """If the requested metric has no known alias, fall back to a real square
    r2r matrix instead of the first (possibly Nx1) key, so it still doesn't
    crash on 'Shape of passed values is (N, 1), indices imply (N, N)'."""
    n = 3
    r2r = np.ones((n, n))
    t2r = np.ones((n, 1))
    mat_path = tmp_path / "sub01_Atlas.tt.gz.Atlas.connectivity.mat"
    scipy_io.savemat(str(mat_path), {"qa t2r": t2r, "qa r2r": r2r})

    extractor = _make_extractor(["some_future_metric"])
    result = extractor.convert_mat_to_csv(mat_path, "Atlas")

    assert result["success"] is True
    assert result["matrix_shape"] == (n, n)


def test_check_connectivity_files_created_accepts_new_bundled_filename(tmp_path):
    """The pre-flight existence check must not mark an atlas as failed just
    because DSI Studio now writes one bundled .mat per atlas instead of one
    file per requested metric."""
    atlas_dir = tmp_path / "results" / "Atlas"
    atlas_dir.mkdir(parents=True)
    bundled = atlas_dir / "sub01.gqi_Atlas.tt.gz.Atlas.connectivity.mat"
    bundled.write_bytes(b"not empty")

    extractor = _make_extractor(["fa"])
    found = extractor._check_connectivity_files_created(tmp_path, "Atlas", "sub01.gqi")

    assert found is True
