"""Tests for MRtrix3 utility functions."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from scripts.utils.mrtrix import (
    parse_lookup_txt,
    read_raw_connectome_matrix,
    write_opticonn_connectivity_csv,
)


def test_parse_lookup_txt_reads_index_name_pairs():
    """Test that parse_lookup_txt reads whitespace-delimited index-name pairs."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("1 Region A\n")
        f.write("2 Region B\n")
        f.write("3 Region C\n")
        tmp_path = Path(f.name)

    try:
        result = parse_lookup_txt(tmp_path)
        assert result == {1: "Region A", 2: "Region B", 3: "Region C"}
    finally:
        tmp_path.unlink()


def test_read_raw_connectome_matrix_loads_whitespace_and_comma_delimited():
    """Test that read_raw_connectome_matrix handles both whitespace and comma delimiters."""
    # Test whitespace-delimited
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("1.0 2.0\n")
        f.write("3.0 4.0\n")
        tmp_path = Path(f.name)

    try:
        result = read_raw_connectome_matrix(tmp_path)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)
    finally:
        tmp_path.unlink()

    # Test comma-delimited
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("1.0,2.0\n")
        f.write("3.0,4.0\n")
        tmp_path = Path(f.name)

    try:
        result = read_raw_connectome_matrix(tmp_path)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)
    finally:
        tmp_path.unlink()


def test_read_raw_connectome_matrix_forces_2d_for_1x1():
    """Test that read_raw_connectome_matrix converts 1-element matrix to 2D."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("42.0\n")
        tmp_path = Path(f.name)

    try:
        result = read_raw_connectome_matrix(tmp_path)
        assert result.ndim == 2
        assert result.shape == (1, 1)
        assert result[0, 0] == 42.0
    finally:
        tmp_path.unlink()


def test_write_opticonn_connectivity_csv_labels_rows_and_columns():
    """Test that write_opticonn_connectivity_csv creates CSV with labeled rows and columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create lookup file
        lookup_path = tmpdir / "lookup.txt"
        lookup_path.write_text("1 Region A\n2 Region B\n")

        # Create matrix file
        matrix_path = tmpdir / "matrix.txt"
        matrix_path.write_text("10.0 20.0\n30.0 40.0\n")

        # Write the CSV
        out_csv = tmpdir / "connectivity.csv"
        write_opticonn_connectivity_csv(matrix_path, lookup_path, out_csv)

        # Verify the CSV was created and has correct content
        assert out_csv.exists()
        content = out_csv.read_text()

        # Check that row and column labels are present
        assert "Region A" in content
        assert "Region B" in content
        # Check that values are present
        assert "10.0" in content or "10" in content
        assert "20.0" in content or "20" in content
        assert "30.0" in content or "30" in content
        assert "40.0" in content or "40" in content
