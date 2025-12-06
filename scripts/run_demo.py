#!/usr/bin/env python3
"""
OptiConn Demo Script
====================

This script demonstrates the full OptiConn pipeline by:
1. Downloading sample data (HCP-YA subjects)
2. Running a short Bayesian Optimization
3. Running a short Cross-Validation Optimization

Usage:
    python scripts/run_demo.py
"""

import os
import sys
import shutil
import subprocess
import requests
import time
from pathlib import Path

# Configuration
DEMO_CONFIG = "configs/demo_config.json"
DATA_DIR = "data/demo_samples"
SUBJECT_IDS = ["100307", "100408", "101107"]
BASE_URL = "https://github.com/data-hcp/lifespan/releases/download/hcp-ya/"


def print_header(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60 + "\n")


def check_dsi_studio():
    print("Checking for DSI Studio...")
    if shutil.which("dsi_studio") is None:
        # Try some common locations on macOS
        common_paths = [
            "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
            "/Applications/DSI Studio.app/Contents/MacOS/dsi_studio",
        ]
        found = False
        for p in common_paths:
            if os.path.exists(p):
                print(f"Found DSI Studio at {p}")
                # Update config or add to PATH?
                # For now, we'll just warn and hope the user has it in PATH or config is set right.
                # Actually, let's update the PATH for this process
                os.environ["PATH"] += os.pathsep + os.path.dirname(p)
                found = True
                break

        if not found:
            print("WARNING: 'dsi_studio' command not found in PATH.")
            print("Please ensure DSI Studio is installed and available in your PATH.")
            print("Or update 'dsi_studio_cmd' in configs/demo_config.json")
            # We continue, as it might be defined in the config as a full path
    else:
        print("DSI Studio found in PATH.")


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="")
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        sys.exit(1)


def prepare_data():
    print_header("Step 1: Preparing Data")
    os.makedirs(DATA_DIR, exist_ok=True)

    for subject_id in SUBJECT_IDS:
        filename = f"{subject_id}.qsdr.fz"
        url = f"{BASE_URL}{filename}"
        dest_path = os.path.join(DATA_DIR, filename)
        download_file(url, dest_path)


def run_bayesian_optimization():
    print_header("Step 2: Running Bayesian Optimization (Demo)")
    print("This will run a short optimization to demonstrate the process.")

    cmd = [
        sys.executable,
        "scripts/bayesian_optimizer.py",
        "--config",
        DEMO_CONFIG,
        "--n-iterations",
        "3",  # Very short run
        "--data-dir",
        DATA_DIR,
        "--output-dir",
        "analysis_results/demo/bayesian",
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
        print("\nBayesian Optimization completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError running Bayesian Optimization: {e}")
        # Don't exit, try the next one


def run_cross_validation():
    print_header("Step 3: Running Cross-Validation Optimization (Demo)")
    print("This will run a short cross-validation to demonstrate the process.")

    cmd = [
        sys.executable,
        "scripts/cross_validation_bootstrap_optimizer.py",
        "--config",
        DEMO_CONFIG,
        "--subjects",
        "3",
        "--data-dir",
        DATA_DIR,
        "--output-dir",
        "analysis_results/demo/cv",
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=os.environ.copy())
        print("\nCross-Validation Optimization completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError running Cross-Validation: {e}")


def main():
    print_header("OptiConn Pipeline Demo")
    print(
        "This script will guide you through a demonstration of the OptiConn pipeline."
    )

    check_dsi_studio()
    prepare_data()

    # Ensure scripts are executable (optional, but good practice)
    # But we are running with python executable, so it's fine.

    run_bayesian_optimization()
    run_cross_validation()

    print_header("Demo Completed")
    print("You can find the results in:")
    print(f"  - {os.path.abspath('analysis_results/demo/bayesian')}")
    print(f"  - {os.path.abspath('analysis_results/demo/cv')}")
    print("\nTo run a full analysis, edit 'configs/production_config.json' and run:")
    print(
        "  python scripts/bayesian_optimizer.py --config configs/production_config.json"
    )
    print("\n")


if __name__ == "__main__":
    main()
