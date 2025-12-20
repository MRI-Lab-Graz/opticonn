#!/usr/bin/env python3
"""
OptiConn MRtrix3 Demo (OpenNeuro Slackline Study)
================================================

This demo demonstrates the OptiConn optimization workflow using the MRtrix3 backend.
It downloads a small subset of the "Slackline" study (ds003138) from OpenNeuro,
skips the heavy preprocessing steps, and runs the Bayesian parameter optimization.

Requirements:
  - openneuro-py (pip install openneuro-py)
  - MRtrix3 installed and on PATH
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def main():
    root = repo_root()
    demo_dir = root / "demo_mrtrix"
    demo_dir.mkdir(exist_ok=True)

    print("=== OptiConn MRtrix3 Demo ===")
    
    # 1. Download data from OpenNeuro
    # We use ds003138 (Slackline study)
    # To save time, we only download a few necessary files for one subject.
    # Note: In a real scenario, you would have QSIPrep/QSIRecon derivatives.
    # For this demo, we simulate the presence of these derivatives.
    
    dataset_id = "ds003138"
    subject = "sub-82KK02101"
    
    print(f"\n[1/3] Downloading subset of {dataset_id} ({subject}) from OpenNeuro...")
    
    # We need:
    # - DWI data (to simulate a bundle if derivatives aren't available)
    # - Or better: download pre-computed derivatives if they exist.
    # Since we want to skip preprocessing, we'll look for derivatives.
    
    try:
        import openneuro
    except ImportError:
        print("Error: openneuro-py not installed. Please run: pip install openneuro-py")
        return 1

    # Download minimal files to demo the discovery and tuning
    # We'll download the raw data and then "mock" the derivatives for the sake of the demo
    # if they are not available on OpenNeuro.
    
    # For the sake of a fast demo, we'll just download the participants.tsv and one T1w
    # to show we can talk to OpenNeuro, then we'll use a small internal mock if available,
    # or instructions on how to run with full data.
    
    # Actually, let's try to download the actual derivatives if they exist.
    # OpenNeuro derivatives are often in a separate 'derivatives' folder.
    
    print(f" Downloading {subject} metadata...")
    openneuro.download(dataset=dataset_id, target_dir=str(demo_dir), 
                      include=[f"{subject}/ses-3/dwi/*", "participants.tsv"],
                      exclude=["*/*.nii.gz"]) # Just download sidecars to be fast for demo
    
    print(f"\n[2/3] Simulating tracking-ready bundle (skipping 5h preprocessing)...")
    # In a real run, you would have run QSIPrep + QSIRecon.
    # Here we create a mock directory structure that opticonn mrtrix-discover expects.
    
    derivatives_dir = demo_dir / "derivatives"
    qsirecon_dir = derivatives_dir / "qsirecon"
    qsiprep_dir = derivatives_dir / "qsiprep"
    
    subj_recon = qsirecon_dir / subject / "ses-3" / "dwi"
    subj_prep = qsiprep_dir / subject / "ses-3" / "dwi"
    
    subj_recon.mkdir(parents=True, exist_ok=True)
    subj_prep.mkdir(parents=True, exist_ok=True)
    
    # Create dummy files to satisfy the discovery script
    # (In a real demo, these would be the actual outputs of QSIRecon)
    (subj_recon / f"{subject}_ses-3_space-T1w_desc-preproc_model-msmtcsd_diffp-WM_dwimap.mif.gz").touch()
    (subj_recon / f"{subject}_ses-3_space-T1w_desc-hsvs_probseg.nii.gz").touch()
    (subj_recon / f"{subject}_ses-3_space-T1w_desc-aparcaseg_dseg.nii.gz").touch()
    (subj_recon / f"{subject}_ses-3_space-T1w_desc-aparcaseg_lookup.txt").write_text("1 Left-Cerebral-Exterior\n2 Left-Cerebral-White-Matter\n")
    (subj_prep / f"{subject}_ses-3_space-T1w_desc-brain_mask.nii.gz").touch()

    print(f" Created mock derivatives in {derivatives_dir}")

    # 3. Run OptiConn Discovery
    print(f"\n[3/3] Running OptiConn Discovery and Bayesian Optimization (Dry-Run)...")
    
    bundle_json = demo_dir / "mrtrix_bundle.json"
    
    discover_cmd = [
        sys.executable, str(root / "opticonn.py"), "mrtrix-discover",
        "--derivatives-dir", str(derivatives_dir),
        "--subject", subject,
        "--session", "ses-3",
        "-o", str(bundle_json)
    ]
    
    print(f" Running: {' '.join(discover_cmd)}")
    # We'll run this for real as it just creates a JSON
    subprocess.run(discover_cmd, check=True)
    
    # Now run the Bayesian optimization (Dry-Run to show it works without needing MRtrix3 installed)
    tune_cmd = [
        sys.executable, str(root / "opticonn.py"), "tune-bayes",
        "--backend", "mrtrix",
        "-i", str(bundle_json),
        "-o", str(demo_dir / "tuning_results"),
        "--n-iterations", "2",
        "--dry-run"
    ]
    
    print(f"\n Running: {' '.join(tune_cmd)}")
    subprocess.run(tune_cmd, check=True)
    
    print("\n=== Demo Completed Successfully! ===")
    print(f"The demo simulated the discovery of a tracking bundle and initiated")
    print(f"a Bayesian optimization run. In a real environment with MRtrix3,")
    print(f"you would remove --dry-run to perform the actual tractography.")

if __name__ == "__main__":
    sys.exit(main())
