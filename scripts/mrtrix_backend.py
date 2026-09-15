#!/usr/bin/env python3
"""
MRtrix3 tractography + connectome extraction for OptiConn.

Input: one folder per subject with
  wmfod.mif        white-matter FOD (seeding and tracking)
  <atlas>.mif      parcellation with integer node labels, same space as wmfod.mif
  5tt.mif          optional; enables anatomically constrained tractography

Output per subject and atlas:
  <output>/<subject>/results/<atlas>/<subject>_<atlas>.count..end.connectivity.mat
(the layout scripts.reliability.collect_matrices reads).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io

from scripts.sweep_utils import find_subject_inputs

# config key in tracking_parameters -> tckgen option; 0/None means "use the MRtrix3 default"
_TCKGEN_OPTIONS = (
    ("cutoff", "-cutoff"),
    ("angle", "-angle"),
    ("step", "-step"),
    ("min_length", "-minlength"),
    ("max_length", "-maxlength"),
)


def build_commands(subject_dir: Path, atlas: str, cfg: dict, work_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    tp = cfg.get("tracking_parameters") or {}
    fod = subject_dir / "wmfod.mif"
    tck = work_dir / f"{atlas}.tck"
    threads = str(cfg.get("thread_count", 4))
    tckgen = [
        "tckgen", str(fod), str(tck),
        "-algorithm", str(tp.get("algorithm", "iFOD2")),
        "-select", str(cfg.get("tract_count", 1000000)),
        "-seed_dynamic", str(fod),
        "-nthreads", threads, "-quiet", "-force",
    ]
    for key, option in _TCKGEN_OPTIONS:
        if tp.get(key):
            tckgen += [option, str(tp[key])]
    act = subject_dir / "5tt.mif"
    if act.exists():
        tckgen += ["-act", str(act)]
    tck2connectome = [
        "tck2connectome", str(tck), str(subject_dir / f"{atlas}.mif"), str(work_dir / f"{atlas}.connectome.csv"),
        "-symmetric", "-zero_diagonal",
        "-nthreads", threads, "-quiet", "-force",
    ]
    return tckgen, tck2connectome, {"MRTRIX_RNG_SEED": str(tp.get("random_seed", 0))}


def matrix_csv_to_mat(csv_path: Path, mat_path: Path) -> None:
    """Numeric matrix text file (comma or whitespace delimited, '#' comments) -> .mat with `connectivity`."""
    first = next(line for line in Path(csv_path).read_text().splitlines() if line.strip() and not line.startswith("#"))
    matrix = np.loadtxt(csv_path, delimiter="," if "," in first else None)
    scipy.io.savemat(str(mat_path), {"connectivity": matrix})


def run_subject(subject_dir: Path, cfg: dict, out_dir: Path) -> None:
    subject = subject_dir.name
    for atlas in cfg["atlases"]:
        parcellation = subject_dir / f"{atlas}.mif"
        if not parcellation.exists():
            raise FileNotFoundError(f"missing parcellation {parcellation}")
        results = out_dir / subject / "results" / atlas
        results.mkdir(parents=True, exist_ok=True)
        tckgen, tck2connectome, env = build_commands(subject_dir, atlas, cfg, results)
        for cmd in (tckgen, tck2connectome):
            print(f"🚀 {' '.join(cmd)}")
            subprocess.run(cmd, check=True, env={**os.environ, **env})
        matrix_csv_to_mat(results / f"{atlas}.connectome.csv", results / f"{subject}_{atlas}.count..end.connectivity.mat")
        (results / f"{atlas}.tck").unlink(missing_ok=True)  # streamline files are large and not used downstream


def main() -> int:
    ap = argparse.ArgumentParser(description="MRtrix3 tractography + connectome extraction for OptiConn")
    ap.add_argument("--data-dir", required=True, help="Folder of subject folders (wmfod.mif, <atlas>.mif, optional 5tt.mif)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=True)
    if len(sys.argv) == 1:
        ap.print_help()
        return 0
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    subjects = find_subject_inputs(Path(args.data_dir), "mrtrix3")
    if not subjects:
        print(f"❌ No subject folders with wmfod.mif in {args.data_dir}")
        return 1
    failed = []
    for subject_dir in subjects:
        print(f"🧠 {subject_dir.name}")
        try:
            run_subject(subject_dir, cfg, Path(args.output))
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ {subject_dir.name}: {e}")
            failed.append(subject_dir.name)
    print(f"✅ {len(subjects) - len(failed)}/{len(subjects)} subjects processed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
