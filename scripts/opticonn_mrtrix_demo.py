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
import random
import pandas as pd
from pathlib import Path

# ANSI Colors
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def main():
    root = repo_root()
    demo_dir = root / "demo_mrtrix"
    
    # Clean up previous demo runs to ensure a fresh start
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(exist_ok=True)

    print(f"{CYAN}{BOLD}=== OptiConn MRtrix3 Official Demo ==={RESET}")
    print(f"{CYAN}This demo showcases the automated BIDS-aware optimization workflow.{RESET}")
    
    # 1. Data Acquisition from OpenNeuro
    dataset_id = "ds003138" # Slackline study
    doi = "10.18112/openneuro.ds003138.v1.1.0"
    
    print(f"\n{GREEN}[1/4] Data Acquisition{RESET}")
    print(f" Connecting to OpenNeuro (DOI: {doi})...")
    
    try:
        import openneuro
    except ImportError:
        print(f"{RED}Error: openneuro-py not installed. Please run: pip install openneuro-py{RESET}")
        return 1

    # Download participants.tsv first to pick a random subject
    print(f" Fetching study metadata...")
    openneuro.download(dataset=dataset_id, target_dir=str(demo_dir), 
                      include=["participants.tsv"])
    
    participants_file = demo_dir / "participants.tsv"
    if not participants_file.exists():
        print(f"{RED}Error: Failed to download participants.tsv{RESET}")
        return 1
        
    df = pd.read_csv(participants_file, sep='\t')
    subjects = df['participant_id'].tolist()
    subject = random.choice(subjects)
    
    print(f" {GREEN}✔{RESET} Found {len(subjects)} participants. Selected random subject: {BOLD}{subject}{RESET}")
    
    # Download minimal sidecars for the selected subject
    print(f" Downloading BIDS sidecars for {subject}...")
    openneuro.download(dataset=dataset_id, target_dir=str(demo_dir), 
                      include=[f"{subject}/ses-3/dwi/*.json"],
                      exclude=["*.nii.gz", "*.mif"])
    
    # 2. Fast-forward Preprocessing (Derivative Simulation)
    print(f"\n{GREEN}[2/4] Simulating Preprocessed Derivatives{RESET}")
    print(f" {YELLOW}Note: Real preprocessing (QSIPrep/QSIRecon) takes ~5-10 hours.{RESET}")
    print(f" Fast-forwarding to the optimization stage by mocking tracking-ready files...")
    
    derivatives_dir = demo_dir / "derivatives"
    qsirecon_dir = derivatives_dir / "qsirecon"
    qsiprep_dir = derivatives_dir / "qsiprep"
    
    # We assume ses-3 for this dataset
    session = "ses-3"
    subj_recon_dwi = qsirecon_dir / subject / session / "dwi"
    subj_recon_anat = qsirecon_dir / subject / session / "anat"
    subj_prep_dwi = qsiprep_dir / subject / session / "dwi"
    
    subj_recon_dwi.mkdir(parents=True, exist_ok=True)
    subj_recon_anat.mkdir(parents=True, exist_ok=True)
    subj_prep_dwi.mkdir(parents=True, exist_ok=True)
    
    # Create dummy files to satisfy the discovery script
    (subj_recon_dwi / f"{subject}_{session}_label-WM_dwimap.mif.gz").touch()
    (subj_recon_anat / f"{subject}_{session}_space-T1w_seg-hsvs_probseg.nii.gz").touch()
    (subj_recon_dwi / f"{subject}_{session}_seg-desikan_dseg.nii.gz").touch()
    (subj_recon_dwi / f"{subject}_{session}_seg-desikan_dseg.txt").write_text("1 Left-Cerebral-Exterior\n2 Left-Cerebral-White-Matter\n")
    (subj_prep_dwi / f"{subject}_{session}_space-T1w_desc-brain_mask.nii.gz").touch()

    print(f" {GREEN}✔{RESET} Mock BIDS derivatives structure created in {derivatives_dir.relative_to(root)}")

    # 3. Bundle Discovery
    print(f"\n{GREEN}[3/4] Automated Bundle Discovery{RESET}")
    bundle_json = demo_dir / "mrtrix_bundle.json"
    
    discover_cmd = [
        sys.executable, str(root / "opticonn.py"), "mrtrix-discover",
        "--derivatives-dir", str(derivatives_dir),
        "--subject", subject,
        "--session", session,
        "--atlas", "desikan",
        "-o", str(bundle_json)
    ]
    
    print(f" Running: {CYAN}{' '.join(discover_cmd[2:])}{RESET}")
    subprocess.run(discover_cmd, check=True)
    print(f" {GREEN}✔{RESET} Discovered tracking-ready bundle saved to {bundle_json.relative_to(root)}")

    # 4. Bayesian Optimization (Running with Mock Binaries)
    print(f"\n{GREEN}[4/4] Bayesian Parameter Optimization{RESET}")
    print(f" {CYAN}Setting up mock MRtrix environment for a lightweight 'real' run...{RESET}")
    
    # Create mock MRtrix binaries to allow a "real" run of the optimization logic
    # without requiring valid data or MRtrix3 installation.
    mock_bin_dir = demo_dir / "mock_bin"
    mock_bin_dir.mkdir(exist_ok=True)
    
    for tool in ["tckgen", "tcksift2", "tck2connectome"]:
        script_path = mock_bin_dir / tool
        with open(script_path, "w") as f:
            f.write("#!/usr/bin/env python3\n")
            f.write("import sys\n")
            f.write("from pathlib import Path\n")
            if tool == "tckgen":
                f.write("if len(sys.argv) > 2: Path(sys.argv[2]).touch()\n")
            elif tool == "tcksift2":
                f.write("if len(sys.argv) > 3: Path(sys.argv[3]).touch()\n")
            elif tool == "tck2connectome":
                f.write("import numpy as np\n")
                f.write("if len(sys.argv) > 3:\n")
                f.write("    out_path = Path(sys.argv[3])\n")
                f.write("    mat = np.random.rand(2, 2)\n")
                f.write("    np.savetxt(out_path, mat)\n")
        script_path.chmod(0o755)
    
    # Add mock_bin to PATH to intercept MRtrix calls
    os.environ["PATH"] = str(mock_bin_dir) + os.pathsep + os.environ.get("PATH", "")
    
    tune_cmd = [
        sys.executable, str(root / "opticonn.py"), 
        "--backend", "mrtrix",
        "tune-bayes",
        "-i", str(bundle_json),
        "-o", str(demo_dir / "tuning_results"),
        "--subject", subject,
        "--n-iterations", "3"
    ]
    
    print(f" Running Bayesian optimization (3 iterations)...")
    print(f" Command: {CYAN}{' '.join(tune_cmd[2:])}{RESET}")
    
    # Run the actual optimization logic
    subprocess.run(tune_cmd, check=True)
    
    print(f"\n{CYAN}{BOLD}=== Demo Completed Successfully! ==={RESET}")
    print(f" The demo successfully executed the full Bayesian optimization logic.")
    print(f" All relevant output files have been generated in {demo_dir.relative_to(root)}/tuning_results")
    print(f" You can now explore the results or run 'opticonn select' on this output.")

if __name__ == "__main__":
    sys.exit(main())


