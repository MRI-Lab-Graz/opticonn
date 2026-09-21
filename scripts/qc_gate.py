"""Flag scans whose preprocessed DWI degraded during preprocessing.

qsiprep writes `*_desc-image_qc.tsv` per session. `t1post_dwi_contrast` is the
contrast of the *final* preprocessed DWI; in study 129 the two scans with odd
tractography (sub-043_ses-1, sub-096_ses-2) were the two lowest in the cohort
(1.24-1.27 vs a median of 2.04) although their raw contrast was normal. A robust
z-score (median/MAD) flags such cohort outliers without assuming a threshold in
absolute units. Flags are advisory: write them out, review, then list scan ids
under `data_selection.exclude_scans` in a wave config to keep them out of tuning.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRIC = "t1post_dwi_contrast"
Z_THRESHOLD = 3.5
MIN_SCANS = 10  # a MAD over fewer scans is not a meaningful cohort estimate
_SCAN_RE = re.compile(r"(sub-[A-Za-z0-9]+)(?:_(ses-[A-Za-z0-9]+))?")


def scan_id(subject: str, session: str | None) -> str:
    return f"{subject}_{session}" if session else subject


def load_qc(qc_root: Path, metric: str = METRIC) -> pd.Series:
    """Metric per scan id from every `*image_qc.tsv` under `qc_root`."""
    values: dict[str, float] = {}
    for tsv in sorted(Path(qc_root).rglob("*image_qc.tsv")):
        if ".git" in tsv.parts or not tsv.is_file():
            continue  # unfetched annex symlinks resolve to nothing
        row = pd.read_csv(tsv, sep="\t").iloc[0]
        if metric not in row.index:
            continue
        m = _SCAN_RE.search(tsv.name)
        if m:
            values[scan_id(m.group(1), m.group(2))] = float(row[metric])
    return pd.Series(values, dtype=float).sort_index()


def flag_low_outliers(values: pd.Series, z: float = Z_THRESHOLD) -> pd.DataFrame:
    """Rows for scans whose robust z is below -z (low contrast only)."""
    if len(values) < MIN_SCANS:
        logging.warning("QC gate: only %d scans (< %d); not flagging", len(values), MIN_SCANS)
        return pd.DataFrame(columns=["scan", "value", "robust_z"])
    med = values.median()
    mad = 1.4826 * (values - med).abs().median()
    if mad == 0:
        return pd.DataFrame(columns=["scan", "value", "robust_z"])
    rz = (values - med) / mad
    out = pd.DataFrame({"scan": values.index, "value": values.values, "robust_z": rz.values})
    return out[out.robust_z < -z].sort_values("robust_z").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("qc_root", type=Path, help="qsiprep derivatives dir containing *image_qc.tsv")
    ap.add_argument("-o", "--output", type=Path, help="write flagged scans to this CSV")
    ap.add_argument("--metric", default=METRIC)
    ap.add_argument("--z", type=float, default=Z_THRESHOLD)
    args = ap.parse_args(argv)
    values = load_qc(args.qc_root, args.metric)
    flagged = flag_low_outliers(values, args.z)
    print(f"{len(values)} scans read; {len(flagged)} flagged ({args.metric}, robust z < -{args.z})")
    if len(flagged):
        print(flagged.to_string(index=False))
    if args.output:
        flagged.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
