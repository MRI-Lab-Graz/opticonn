#!/usr/bin/env python3
"""
OptiConn Full Demo
==================

This script demonstrates the full capabilities of the OptiConn pipeline.
It performs the following steps:
1. Downloads sample DSI Studio data (HCP subject).
2. Prepares a demo dataset (simulating multiple subjects).
3. Runs Bayesian Optimization (The "Smart" Way).
4. Runs Cross-Validation Optimization (The "Robust" Way).

Usage:
    python scripts/run_full_demo.py
"""

import os
import sys
import shutil
import json
import subprocess
import time
import requests
from pathlib import Path

# Configuration
DEMO_DIR = Path("demo_workspace")
DATA_DIR = DEMO_DIR / "data"
RESULTS_DIR = DEMO_DIR / "results"
CONFIG_DIR = DEMO_DIR / "configs"
SAMPLE_URL = "https://github.com/frankyeh/DSI-Studio-Website/raw/master/data/100307.qsdr.fz"  # Smaller sample if available, or use the one from before
# The previous URL was https://github.com/data-hcp/lifespan/releases/download/hcp-ya/100307.qsdr.fz which might be large.
# Let's use a known small sample or the one we know works.
# Frank Yeh often provides a demo file. Let's stick to the one we saw or a reliable one.
# The one in download_mousley_data.py is likely fine.
SAMPLE_URL = (
    "https://github.com/data-hcp/lifespan/releases/download/hcp-ya/100307.qsdr.fz"
)


def setup_directories():
    print(f"Creating demo workspace at {DEMO_DIR}...")
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)


def download_data():
    print("\n[1/4] Setting up demo data...")
    target_file = DATA_DIR / "subject_01.fz"

    print(f"  Downloading sample subject to {target_file}...")
    try:
        # Check if we already have it in the main data folder to save time/bandwidth
        cache_path = Path("data/fib_samples/100307.qsdr.fz")
        if cache_path.exists():
            print("  Found cached file, copying...")
            shutil.copy(cache_path, target_file)
        else:
            # Download
            r = requests.get(SAMPLE_URL, stream=True)
            r.raise_for_status()
            with open(target_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"  Error downloading data: {e}")
        print(
            "  Please ensure you have internet access or place a .fz file in demo_workspace/data/subject_01.fz manually."
        )
        return False

    # Simulate 3 subjects for Cross-Validation
    print("  Simulating cohort (copying subject to create n=3 dataset)...")
    shutil.copy(target_file, DATA_DIR / "subject_02.fz")
    shutil.copy(target_file, DATA_DIR / "subject_03.fz")
    print(f"  Data ready in {DATA_DIR}")
    return True


def create_demo_config():
    print("\n[2/4] Creating demo configuration...")

    # Detect DSI Studio
    dsi_studio_cmd = "dsi_studio"
    # Try common paths on macOS
    common_paths = [
        "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
        str(Path.home() / "Applications/dsi_studio.app/Contents/MacOS/dsi_studio"),
    ]
    for p in common_paths:
        if os.path.exists(p):
            dsi_studio_cmd = p
            break

    print(f"  Using DSI Studio executable: {dsi_studio_cmd}")

    config = {
        "comment": "Demo Configuration",
        "dsi_studio_cmd": dsi_studio_cmd,
        "atlases": ["AAL3"],  # Keep it simple for demo
        "connectivity_values": ["count", "fa", "qa"],
        "tract_count": 5000,  # Low count for speed
        "thread_count": 4,
        "tracking_parameters": {
            "method": 0,
            "otsu_threshold": 0.6,
            "fa_threshold": 0.0,
            "threshold_index": "qa",
            "min_length": 30,
            "max_length": 300,
            "track_voxel_ratio": 2.0,
            "check_ending": 0,
            "tip_iteration": 0,
            "random_seed": 1234,
            "dt_threshold": 0.0,
        },
        "connectivity_options": {
            "connectivity_type": "pass",
            "connectivity_threshold": 0.001,
            "connectivity_output": "matrix,measure",
        },
        "sweep_parameters": {
            "description": "Demo Sweep",
            "sampling": {
                "method": "lhs",
                "n_samples": 3,  # Only 3 iterations for speed
                "random_seed": 42,
            },
            # Small ranges for demo
            "tract_count_range": [5000, 10000],
            "fa_threshold_range": [0.0, 0.1],
            "turning_angle_range": [40.0, 60.0],
            "step_size_range": [0.5, 1.5],
            "min_length_range": [20, 50],
            "connectivity_threshold_range": [0.001, 0.01],
        },
    }

    config_path = CONFIG_DIR / "demo_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Config saved to {config_path}")
    return config_path


def run_bayesian_demo(config_path):
    print("\n[3/4] Running Bayesian Optimization Demo...")
    print("  Goal: Efficiently find optimal parameters using Gaussian Processes.")

    cmd = [
        sys.executable,
        "-m",
        "scripts.bayesian_optimizer",
        "--data-dir",
        str(DATA_DIR),
        "--output-dir",
        str(RESULTS_DIR / "bayesian"),
        "--config",
        str(config_path),
        "--n-iterations",
        "3",
        "--verbose",
    ]

    print(f"  Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("  Bayesian Demo Completed Successfully!")
    except subprocess.CalledProcessError:
        print("  Bayesian Demo Failed.")


def run_cv_demo(config_path):
    print("\n[4/4] Running Cross-Validation Demo...")
    print("  Goal: Validate parameter stability across two independent waves.")

    cmd = [
        sys.executable,
        "-m",
        "scripts.cross_validation_bootstrap_optimizer",
        "--data-dir",
        str(DATA_DIR),
        "--output-dir",
        str(RESULTS_DIR / "cross_validation"),
        "--extraction-config",
        str(config_path),
        "--subjects",
        "3",
        "--verbose",
    ]

    print(f"  Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("  Cross-Validation Demo Completed Successfully!")
    except subprocess.CalledProcessError:
        print("  Cross-Validation Demo Failed.")


def main():
    print("==================================================")
    print("       OptiConn Pipeline - End-User Demo")
    print("==================================================")

    setup_directories()

    if not download_data():
        return

    config_path = create_demo_config()

    run_bayesian_demo(config_path)

    run_cv_demo(config_path)

    print("\n==================================================")
    print(" Demo Run Finished.")
    print(f" Results are available in: {RESULTS_DIR.absolute()}")
    print("==================================================")


if __name__ == "__main__":
    main()
