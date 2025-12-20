# OptiConn MRtrix3 Demo Walkthrough

This document provides a detailed walkthrough of the **Official OptiConn MRtrix3 Demo** (`scripts/opticonn_mrtrix_demo.py`). This demo is designed for JOSS reviewers and new users to evaluate the pipeline's core optimization logic in minutes, without requiring hours of dMRI preprocessing.

## Overview

The demo automates a 4-step workflow with informative terminal feedback:
1. **Data Acquisition**: Pulls study metadata from OpenNeuro and selects a participant randomly.
2. **Derivative Simulation**: Mocks the outputs of a QSIRecon/MRtrix workflow (fast-forwarding 5-10 hours of compute).
3. **Bundle Discovery**: Automatically identifies tracking-ready files using BIDS conventions.
4. **Bayesian Optimization**: Runs the parameter search engine (real or simulated).

---

## Prerequisites

- **Python 3.10+**
- **openneuro-py**: `pip install openneuro-py`
- **MRtrix3** (Optional): The demo detects if MRtrix3 is installed. If missing, it runs in simulation mode (`--dry-run`) to show the pipeline logic.

---

## Detailed Steps

### 1. Data Acquisition (OpenNeuro)
The script uses the `openneuro-py` library to communicate with the [OpenNeuro](https://openneuro.org/) platform. It targets the **Slackline study (ds003138)** (DOI: 10.18112/openneuro.ds003138.v1.1.0).

**What happens:**
- Downloads `participants.tsv`.
- **Random Selection**: Picks one subject from the cohort to demonstrate cohort-wide compatibility.
- Downloads minimal BIDS sidecars (`.json`) for the selected subject.

### 2. Derivative Simulation
Tractography requires preprocessed data (FODs, tissue masks, parcellations). In a real-world study, you would run [QSIPrep](https://qsiprep.readthedocs.io/) and [QSIRecon](https://qsiprep.readthedocs.io/en/latest/reconstruction.html).

Since these steps are computationally expensive, the demo **mocks** the expected directory structure:
- **WM FOD**: `sub-ID_ses-3_label-WM_dwimap.mif.gz`
- **ACT Mask**: `sub-ID_ses-3_space-T1w_seg-hsvs_probseg.nii.gz`
- **Parcellation**: `sub-ID_ses-3_seg-desikan_dseg.nii.gz`

### 3. Bundle Discovery (`mrtrix-discover`)
OptiConn includes a discovery tool that scans your derivatives folder to find the files created in Step 2.

**Command executed by the demo:**
```bash
opticonn mrtrix-discover \
    --derivatives-dir demo_mrtrix/derivatives \
    --subject sub-RANDOM \
    --session ses-3 \
    --atlas desikan \
    -o demo_mrtrix/mrtrix_bundle.json
```

### 4. Bayesian Optimization (`tune-bayes`)
This is the heart of OptiConn. It uses Gaussian Process regression to find the tractography parameters that maximize network quality.

**Command executed by the demo:**
```bash
opticonn --backend mrtrix tune-bayes \
    -i demo_mrtrix/mrtrix_bundle.json \
    -o demo_mrtrix/tuning_results \
    --subject sub-RANDOM \
    --n-iterations 3
```

**Execution Strategy:**
To ensure a "slim" and fast experience without requiring valid dMRI data (which is huge), the demo script sets up a **mock MRtrix environment**. It creates lightweight scripts that simulate `tckgen` and `tck2connectome`. This allows the **actual Bayesian optimization logic** to run, iterate, and generate real result files (`.csv`, `.json`) without the computational overhead of real tractography.

---

## Interpreting the Output

After running the demo, you will see the following in the `demo_mrtrix/` directory:
- `mrtrix_bundle.json`: The discovered paths for the subject.
- `derivatives/`: The mocked BIDS structure.
- `tuning_results/`: Contains the optimization logs and the `bayesian_optimization_results.json` file.

## Next Steps

To run a real optimization on your own data:
1. Ensure MRtrix3 is installed.
2. Run `opticonn mrtrix-discover` on your QSIRecon derivatives.
3. Run `opticonn tune-bayes` on the resulting bundle.
4. Use `opticonn select` to promote the best candidate.
5. Use `opticonn apply` to process your entire cohort.

