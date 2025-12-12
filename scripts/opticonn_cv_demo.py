#!/usr/bin/env python3
"""
OptiConn Cross-Validation Demo
==============================

Lightweight driver to run the cross-validation bootstrap optimizer on
small sample data.

Steps:
1) Prepare demo data (download small HCP YA sample, duplicate to n=3 subjects)
2) Run cross-validation bootstrap optimizer (two waves)

Example:
    python scripts/opticonn_cv_demo.py --workspace demo_workspace_cv
"""

from __future__ import annotations

import argparse
import shutil
import sys
import subprocess
from pathlib import Path

import requests

SAMPLE_URL = "https://github.com/data-hcp/lifespan/releases/download/hcp-ya/100307.qsdr.fz"
SAMPLE_SOURCE = "HCP Young Adult sample (data-hcp/lifespan GitHub release)"
DEFAULT_WORKSPACE = Path("demo_workspace_cv")
DEFAULT_CONFIG = Path("configs/demo_config.json")
DEFAULT_BAYES = Path("demo_workspace/results/bayes/bayesian_optimization_results.json")

RESET = "\033[0m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"


def color(text: str, ansi: str) -> str:
    return f"{ansi}{text}{RESET}"


def download_sample(dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(color(f"Sample already present: {dest}", GREEN))
        return dest

    print(color("Downloading demo sample...", CYAN))
    print(f"  Source: {SAMPLE_SOURCE}")
    print(f"  URL: {SAMPLE_URL}")
    try:
        with requests.get(SAMPLE_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(color(f"Download failed: {e}", MAGENTA))
        sys.exit(1)

    print(color(f"Saved sample to {dest}", GREEN))
    return dest


def prepare_data(data_dir: Path, force_download: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    src = data_dir / "subject_01.fz"
    download_sample(src, force=force_download)
    # Duplicate to simulate 3 subjects for cross-validation waves
    for idx in (2, 3):
        dst = data_dir / f"subject_0{idx}.fz"
        if not dst.exists():
            shutil.copy(src, dst)
    print(color(f"Data ready under {data_dir} (3 subjects)", GREEN))


def run_cross_validation(
    workspace: Path,
    config: Path,
    subjects: int,
    max_parallel: int,
    from_bayes: Path | None,
) -> None:
    data_dir = workspace / "data"
    output_dir = workspace / "results" / "cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/cross_validation_bootstrap_optimizer.py",
        "-i",
        str(data_dir),
        "-o",
        str(output_dir),
        "--extraction-config",
        str(config),
        "--subjects",
        str(subjects),
        "--max-parallel",
        str(max_parallel),
        "--verbose",
    ]
    if from_bayes and from_bayes.exists():
        cmd += ["--from-bayes", str(from_bayes)]
        print(color(f"Seeding parameters from Bayesian results: {from_bayes}", CYAN))

    print(color("\n[Cross-validation] Running two-wave bootstrap optimization", YELLOW))
    print(color("Command: " + " ".join(cmd), MAGENTA))
    sys.exit(subprocess.run(cmd).returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OptiConn cross-validation demo")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Workspace directory")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Extraction config to use")
    parser.add_argument(
        "--from-bayes",
        default=str(DEFAULT_BAYES),
        help="Path to bayesian_optimization_results.json to seed cross-validation (optional)",
    )
    parser.add_argument("--subjects", type=int, default=3, help="Subjects per wave (default: 3)")
    parser.add_argument("--max-parallel", type=int, default=1, help="Max parallel combos per wave")
    parser.add_argument("--force-download", action="store_true", help="Redownload sample")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    config = Path(args.config).resolve()
    if not config.exists():
        print(color(f"Config not found: {config}", MAGENTA))
        return 1

    data_dir = workspace / "data"
    prepare_data(data_dir, args.force_download)
    from_bayes = Path(args.from_bayes).resolve() if args.from_bayes else None
    run_cross_validation(workspace, config, args.subjects, args.max_parallel, from_bayes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
