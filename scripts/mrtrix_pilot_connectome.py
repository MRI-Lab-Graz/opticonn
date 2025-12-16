#!/usr/bin/env python3
"""Pilot MRtrix → OptiConn workflow for one subject/session.

What it does:
1) Run MRtrix `tckgen` from a WM FOD image (quick pilot settings)
2) Run `tck2connectome` using a parcellation dseg image
3) Convert the raw connectome matrix into an OptiConn-style
   `*.count.connectivity.csv` with region names as index/columns
4) Compute network measures (via scripts/compute_network_measures_from_connectivity.py)

This is meant as a "first working" path to enable OptiConn QA scoring.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _parse_lookup_txt(path: Path) -> Dict[int, str]:
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


def _read_raw_connectome_matrix(path: Path) -> np.ndarray:
    # MRtrix outputs a plain numeric matrix. It is often space-delimited.
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
    mat = _read_raw_connectome_matrix(raw_matrix_path)
    lookup = _parse_lookup_txt(lookup_txt)

    n = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Connectome matrix must be square; got {mat.shape}")

    labels: List[str] = []
    for i in range(1, n + 1):
        labels.append(lookup.get(i, str(i)))

    df = pd.DataFrame(mat, index=labels, columns=labels)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pilot MRtrix connectome generation and OptiConn export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--fod", type=Path, required=True, help="WM FOD *.mif.gz")
    p.add_argument(
        "--dseg",
        type=Path,
        required=True,
        help="Parcellation dseg image (prefer *.mif.gz)",
    )
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Lookup table mapping integer labels to names (two columns)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory (will create OptiConn-like results structure)",
    )
    p.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., sub-1293171)")
    p.add_argument("--atlas", type=str, default="Brainnetome246Ext", help="Atlas name")

    # Pilot tractography settings
    p.add_argument("--select", type=int, default=50000, help="Number of streamlines")
    p.add_argument("--cutoff", type=float, default=0.06, help="tckgen -cutoff")
    p.add_argument("--angle", type=float, default=45.0, help="tckgen -angle")
    p.add_argument("--minlength", type=float, default=30.0, help="tckgen -minlength")
    p.add_argument("--maxlength", type=float, default=250.0, help="tckgen -maxlength")
    p.add_argument("--nthreads", type=int, default=8, help="Number of threads")

    # Network measures
    p.add_argument(
        "--smallworld",
        action="store_true",
        help="Compute small-worldness(binary) (slower)",
    )

    args = p.parse_args()

    # OptiConn Step01-style layout (minimal)
    results_dir = args.outdir / "01_connectivity" / "pilot" / "theta_000" / "results" / args.atlas
    results_dir.mkdir(parents=True, exist_ok=True)

    tractogram = args.outdir / "01_connectivity" / "pilot" / "theta_000" / "tractogram.tck"
    raw_connectome = args.outdir / "01_connectivity" / "pilot" / "theta_000" / f"{args.subject}_{args.atlas}.count.connectome_raw.csv"
    out_connectivity = results_dir / f"{args.subject}_{args.atlas}.count.connectivity.csv"

    # 1) tckgen
    _run(
        [
            "tckgen",
            str(args.fod),
            str(tractogram),
            "-algorithm",
            "iFOD2",
            "-seed_dynamic",
            str(args.fod),
            "-select",
            str(args.select),
            "-cutoff",
            str(args.cutoff),
            "-angle",
            str(args.angle),
            "-minlength",
            str(args.minlength),
            "-maxlength",
            str(args.maxlength),
            "-nthreads",
            str(args.nthreads),
            "-force",
        ]
    )

    # 2) tck2connectome
    _run(
        [
            "tck2connectome",
            str(tractogram),
            str(args.dseg),
            str(raw_connectome),
            "-assignment_radial_search",
            "2",
            "-zero_diagonal",
            "-symmetric",
            "-nthreads",
            str(args.nthreads),
            "-force",
        ]
    )

    # 3) Convert to OptiConn-style connectivity.csv with region labels
    write_opticonn_connectivity_csv(raw_connectome, args.labels, out_connectivity)
    print(f"Wrote connectivity: {out_connectivity}")

    # 4) Compute network measures (writes alongside connectivity by default)
    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "compute_network_measures_from_connectivity.py"),
        str(out_connectivity),
    ]
    if args.smallworld:
        cmd.append("--smallworld")
    _run(cmd)

    print("Done.")
    print(f"Step01 root: {args.outdir / '01_connectivity'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
