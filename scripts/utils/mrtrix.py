#!/usr/bin/env python3
"""
Utilities for MRtrix3 integration.
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

def parse_lookup_txt(path: Path) -> Dict[int, str]:
    """Parse a lookup table (index name) into a dictionary."""
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = " ".join(parts[1:])
            mapping[idx] = name
    return mapping

def read_raw_connectome_matrix(path: Path) -> np.ndarray:
    """Read a raw MRtrix connectome matrix (space or comma delimited)."""
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"Empty connectome output: {path}")

    # Try whitespace first, then comma
    for delimiter in (None, ","):
        try:
            mat = np.loadtxt(path, delimiter=delimiter)
            if mat.ndim == 1:
                mat = np.atleast_2d(mat)
            return mat
        except Exception:
            continue

    raise ValueError(f"Could not parse connectome matrix: {path}")

def write_opticonn_connectivity_csv(
    raw_matrix_path: Path,
    lookup_txt: Path,
    out_csv: Path,
) -> None:
    """Convert a raw MRtrix connectome matrix to an OptiConn-style CSV."""
    mat = read_raw_connectome_matrix(raw_matrix_path)
    lookup = parse_lookup_txt(lookup_txt)

    n = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Connectome matrix must be square; got {mat.shape}")

    labels: List[str] = []
    for i in range(1, n + 1):
        labels.append(lookup.get(i, str(i)))

    df = pd.DataFrame(mat, index=labels, columns=labels)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
