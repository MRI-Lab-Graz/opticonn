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
import json
import shutil
import sys
import subprocess
from pathlib import Path

import requests

SAMPLE_URL = (
    "https://github.com/data-hcp/lifespan/releases/download/hcp-ya/100307.qsdr.fz"
)
SAMPLE_SOURCE = "HCP Young Adult sample (data-hcp/lifespan GitHub release)"
DEFAULT_WORKSPACE = Path("demo_workspace_cv")
DEFAULT_CONFIG = Path("configs/demo_config.json")
DEFAULT_BAYES_BASE = Path("demo_workspace/results/bayes")

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
    output_dir: Path,
    subjects: int,
    max_parallel: int,
    from_bayes: Path | None,
) -> int:
    data_dir = workspace / "data"
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
    return subprocess.run(cmd).returncode


def write_modality_config(workspace: Path, base_config: Path, modality: str) -> Path:
    cfg_dir = workspace / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["connectivity_values"] = [modality]
    out_path = cfg_dir / f"cv_config_{modality}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OptiConn cross-validation demo")
    parser.add_argument(
        "--workspace", default=str(DEFAULT_WORKSPACE), help="Workspace directory"
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Extraction config to use"
    )
    parser.add_argument(
        "--from-bayes",
        default="",
        help=(
            "Optional: path to bayesian_optimization_results.json to seed cross-validation. "
            "If omitted, the demo seeds per-modality from --bayes-base/<modality>/bayesian_optimization_results.json"
        ),
    )
    parser.add_argument(
        "--bayes-base",
        default=str(DEFAULT_BAYES_BASE),
        help="Base directory containing per-modality bayes outputs (default: demo_workspace/results/bayes)",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["qa", "fa"],
        help="One or more modalities to run CV for (default: qa fa)",
    )
    parser.add_argument(
        "--subjects", type=int, default=3, help="Subjects per wave (default: 3)"
    )
    parser.add_argument(
        "--max-parallel", type=int, default=1, help="Max parallel combos per wave"
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Redownload sample"
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    config = Path(args.config).resolve()
    if not config.exists():
        print(color(f"Config not found: {config}", MAGENTA))
        return 1

    data_dir = workspace / "data"
    prepare_data(data_dir, args.force_download)

    # Back-compat: if user provided an explicit results file, run a single CV seeded from it.
    if args.from_bayes:
        from_bayes = Path(args.from_bayes).resolve()
        out_dir = workspace / "results" / "cv"
        return run_cross_validation(
            workspace, config, out_dir, args.subjects, args.max_parallel, from_bayes
        )

    print(
        color(
            "\n[CV Demo] Running cross-validation seeded from Bayes (per modality)",
            YELLOW,
        )
    )
    bayes_base = Path(args.bayes_base).resolve()
    for m in args.modalities:
        bayes_results = bayes_base / m / "bayesian_optimization_results.json"
        if not bayes_results.exists():
            print(color("Missing Bayesian results to seed CV demo.", MAGENTA))
            print(f"Expected: {bayes_results}")
            return 1

        modality_config = write_modality_config(workspace, config, m)
        out_dir = workspace / "results" / "cv" / m
        print(color(f"\nSeeding CV for modality '{m}'", CYAN))
        rc = run_cross_validation(
            workspace,
            modality_config,
            out_dir,
            args.subjects,
            args.max_parallel,
            bayes_results,
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
