#!/usr/bin/env python3
"""
OptiConn Quick Demo
===================

Lightweight, three-step demo aligned with the main workflow:
1) tune-bayes on a tiny sample
2) select the best candidate
3) apply to the same sample

Usage examples:
    python scripts/opticonn_demo.py --step all
    python scripts/opticonn_demo.py --step 1 --n-iterations 4

Notes:
- Downloads a small DSI Studio sample from the public HCP YA mirror if missing.
- Uses configs/demo_config.json by default.
- Keeps outputs under <workspace>/results.
"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

import requests

SAMPLE_URL = "https://github.com/data-hcp/lifespan/releases/download/hcp-ya/100307.qsdr.fz"
SAMPLE_SOURCE = (
    "HCP Young Adult sample (public mirror: data-hcp/lifespan GitHub release)"
)

RESET = "\033[0m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"


def color(text: str, ansi: str) -> str:
    return f"{ansi}{text}{RESET}"
DEFAULT_CONFIG = Path("configs/demo_config.json")
DEFAULT_WORKSPACE = Path("demo_workspace")


def download_sample(data_dir: Path, force: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "subject_01.fz"
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
        print(f"Download failed: {e}")
        sys.exit(1)

    print(color(f"Saved sample to {dest}", GREEN))
    return dest


def run(cmd: list[str]) -> None:
    print(color(f"Running: {' '.join(cmd)}", MAGENTA))
    subprocess.run(cmd, check=True)


def step1_tune_bayes(workspace: Path, config: Path, n_iterations: int) -> Path:
    data_dir = workspace / "data"
    download_sample(data_dir)

    output_dir = workspace / "results" / "bayes"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "opticonn.py",
        "tune-bayes",
        "-i",
        str(data_dir),
        "-o",
        str(output_dir),
        "--config",
        str(config),
        "--n-iterations",
        str(n_iterations),
        "--sample-subjects",
    ]
    print(color("\n[Step 1] Bayesian tuning", YELLOW))
    run(cmd)
    return output_dir / "bayesian_optimization_results.json"


def step2_select(config_path: Path) -> None:
    cmd = [
        sys.executable,
        "opticonn.py",
        "select",
        "-i",
        str(config_path),
    ]
    print(color("\n[Step 2] Select best candidate", YELLOW))
    run(cmd)


def build_apply_config(results_path: Path, base_config: Path, output_dir: Path) -> Path:
    """Create an apply-ready config that passes validation.

    The raw Bayesian results file lacks required fields. We merge the
    best parameters back into the original config (atlases, metrics,
    dsi_studio_cmd) and write a temporary config for the apply step.
    """

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        best_params = results.get("best_parameters")
    except Exception as e:
        print(f"Failed to load Bayesian results {results_path}: {e}")
        return results_path

    if not best_params:
        print("No best_parameters found in Bayesian results; falling back to raw file.")
        return results_path

    try:
        with open(base_config, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
    except Exception as e:
        print(f"Failed to load base config {base_config}: {e}")
        return results_path

    merged_cfg = dict(base_cfg)
    merged_cfg["best_parameters"] = best_params

    output_dir.mkdir(parents=True, exist_ok=True)
    apply_cfg_path = output_dir / "apply_config.json"
    with open(apply_cfg_path, "w", encoding="utf-8") as f:
        json.dump(merged_cfg, f, indent=2)

    return apply_cfg_path


def step3_apply(workspace: Path, optimal_config: Path, base_config: Path) -> None:
    data_dir = workspace / "data"
    output_dir = workspace / "results" / "apply"
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_cfg = build_apply_config(optimal_config, base_config, output_dir)
    print(color("\n[Step 3] Apply best parameters", YELLOW))
    print(f"Applying with config: {apply_cfg}")

    cmd = [
        sys.executable,
        "opticonn.py",
        "apply",
        "-i",
        str(data_dir),
        "--optimal-config",
        str(apply_cfg),
        "-o",
        str(output_dir),
    ]
    run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OptiConn quick demo")
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which demo step to run (default: all)",
    )
    parser.add_argument(
        "--workspace",
        default=str(DEFAULT_WORKSPACE),
        help="Workspace directory for demo data and results",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Config file to use for demo runs",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=4,
        help="Number of Bayesian iterations for the demo",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload sample data even if already present",
    )

    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    config = Path(args.config).resolve()
    if not config.exists():
        print(f"Config not found: {config}")
        return 1

    workspace.mkdir(parents=True, exist_ok=True)

    # Allow manual force download before any step
    if args.force_download:
        download_sample(workspace / "data", force=True)

    if args.step in ("1", "all"):
        bayes_result = step1_tune_bayes(workspace, config, args.n_iterations)
    else:
        bayes_result = workspace / "results" / "bayes" / "bayesian_optimization_results.json"

    if args.step in ("2", "all"):
        if not bayes_result.exists():
            print(f"Missing Bayesian results for selection: {bayes_result}")
            return 1
        step2_select(bayes_result)

    if args.step in ("3", "all"):
        if not bayes_result.exists():
            print(f"Missing optimal config for apply: {bayes_result}")
            return 1
        step3_apply(workspace, bayes_result, config)

    print("Demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
