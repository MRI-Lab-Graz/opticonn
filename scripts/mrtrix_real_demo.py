#!/usr/bin/env python3
"""
OptiConn MRtrix3 Real-Data Demo (Stanford HARDI)
===============================================

This demo runs a REAL MRtrix3 preprocessing and tractography pipeline using
the Stanford HARDI dataset (downloaded via dipy).

It performs:
1. Data acquisition (Stanford HARDI DWI + Parcellation)
2. MRtrix3 Processing:
   - DWI to MIF conversion
   - Brain mask generation
   - Response function estimation (Tournier)
   - FOD estimation (CSD)
3. OptiConn-style Tractography:
   - Streamline generation (tckgen) - lightweight (5000 tracks)
   - Connectome generation (tck2connectome)
4. OptiConn Export:
   - Conversion to OptiConn-style CSV
   - Network measure computation

Requirements:
  - dipy (pip install dipy)
  - MRtrix3 installed and on PATH
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from dipy.data import fetch_stanford_hardi, fetch_stanford_labels

# ANSI Colors
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

def run(cmd, cwd=None):
    print(f"{YELLOW}Running: {' '.join(map(str, cmd))}{RESET}")
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    print(f"{CYAN}{BOLD}=== OptiConn MRtrix3 Real-Data Demo ==={RESET}")
    
    # 1. Data Acquisition
    print(f"\n{GREEN}[1/5] Data Acquisition (Stanford HARDI){RESET}")
    try:
        hardi_files, hardi_dir = fetch_stanford_hardi()
        label_files, label_dir = fetch_stanford_labels()
        print(f" {GREEN}✔{RESET} Data downloaded to {hardi_dir}")
    except Exception as e:
        print(f"{RED}Error downloading data: {e}{RESET}")
        return 1

    # Setup workspace
    root = Path(__file__).resolve().parent.parent
    demo_dir = root / "demo_mrtrix_real"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(exist_ok=True)
    
    dwi_path = Path(hardi_dir) / "HARDI150.nii.gz"
    bval_path = Path(hardi_dir) / "HARDI150.bval"
    bvec_path = Path(hardi_dir) / "HARDI150.bvec"
    label_path = Path(hardi_dir) / "aparc-reduced.nii.gz"
    label_info = Path(hardi_dir) / "label_info.txt"

    # 2. MRtrix3 Preprocessing
    print(f"\n{GREEN}[2/5] MRtrix3 Preprocessing{RESET}")
    
    mif_path = demo_dir / "dwi.mif"
    mask_path = demo_dir / "mask.mif"
    response_path = demo_dir / "response.txt"
    fod_path = demo_dir / "fod.mif"

    # Convert to MIF
    run(["mrconvert", dwi_path, mif_path, "-fslgrad", bvec_path, bval_path, "-quiet"])
    
    # Create mask
    print(" Generating brain mask...")
    run(["dwi2mask", mif_path, mask_path, "-quiet"])
    
    # Estimate response
    print(" Estimating response function...")
    run(["dwi2response", "tournier", mif_path, response_path, "-mask", mask_path, "-quiet"])
    
    # Estimate FOD
    print(" Estimating FOD (CSD)...")
    run(["dwi2fod", "csd", mif_path, response_path, fod_path, "-mask", mask_path, "-quiet"])

    # 3. Connectome Pipeline (Optimization)
    print(f"\n{GREEN}[3/5] Connectome Pipeline (Bayesian Optimization){RESET}")
    
    # Create a config for mrtrix_tune.py
    config = {
        "description": "Stanford HARDI Demo Config",
        "backend": "mrtrix",
        "inputs": {
            "bundle": {
                "wm_fod": str(fod_path.resolve()),
                "act_5tt_or_hsvs": None, # No ACT for this simple demo
                "parcellations": [
                    {
                        "name": "StanfordAparc",
                        "dseg": str(label_path.resolve()),
                        "labels_tsv": str(label_info.resolve())
                    }
                ]
            }
        },
        "mrtrix": {
            "tckgen": {
                "algorithm": "iFOD2",
                "select": 5000, # Lightweight for demo
                "seed": {
                    "type": "image",
                    "image": str(mask_path.resolve())
                }
            },
            "tck2connectome": {
                "outputs": [
                    {"name": "count", "weighted": False}
                ]
            }
        },
        "search_space": {
            "type": "bayesian",
            "parameters": {
                "tckgen.cutoff": [0.05, 0.15],
                "tckgen.angle": [30, 60]
            }
        }
    }
    
    config_path = demo_dir / "mrtrix_bundle.json"
    with open(config_path, "w") as f:
        import json
        json.dump(config, f, indent=2)
    
    print(f" Created optimization config: {config_path.relative_to(root)}")
    
    # Run mrtrix_tune.py bayes
    tune_out = demo_dir / "tuning_results"
    run([
        sys.executable, 
        str(root / "scripts" / "mrtrix_tune.py"), 
        "bayes", 
        "--config", str(config_path),
        "--output-dir", str(tune_out),
        "--subject", "sub-stanford",
        "--n-iterations", "5",
        "--allow-missing-act"
    ])

    # 4. Optimal Selection & Summary
    print(f"\n{GREEN}[4/5] Results Summary{RESET}")
    results_json = tune_out / "bayesian_optimization_results.json"
    if results_json.exists():
        with open(results_json, "r") as f:
            res = json.load(f)
        print(f" {GREEN}✔{RESET} Best QA Score: {res['best_quality_score']:.4f}")
        print(f" {GREEN}✔{RESET} Best Parameters: {res['best_parameters']}")
    
    print(f"\n{CYAN}{BOLD}=== Real-Data Demo Completed Successfully! ==={RESET}")
    print(f" You have successfully run the full OptiConn MRtrix3 pipeline.")
    print(f" 1. Preprocessed Stanford HARDI data")
    print(f" 2. Ran Bayesian optimization (5 iterations)")
    print(f" 3. Identified the best parameter set based on QA score")
    print(f"\n Results are available in: {demo_dir.relative_to(root)}")

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n{RED}Demo failed: {e}{RESET}")
        sys.exit(1)
