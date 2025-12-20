# OptiConn MRtrix3 Demo Walkthrough

This document provides a detailed walkthrough of the **OptiConn MRtrix3 Demo** (`scripts/opticonn_mrtrix_demo.py`). This demo is designed for JOSS reviewers and new users to evaluate the pipeline's core optimization logic in minutes, without requiring hours of dMRI preprocessing.

## Overview

The demo automates a 4-step workflow:
1. **Data Acquisition**: Downloads metadata from OpenNeuro.
2. **Derivative Simulation**: Mocks the outputs of a QSIRecon/MRtrix workflow.
3. **Bundle Discovery**: Automatically identifies tracking-ready files.
4. **Bayesian Optimization**: Runs a dry-run of the parameter search engine.

---

## Prerequisites

- **Python 3.10+**
- **openneuro-py**: `pip install openneuro-py`
- **MRtrix3** (Optional for dry-run): The demo runs with `--dry-run` by default, so you can see the commands even if MRtrix3 is not installed.

---

## Detailed Steps

### 1. Data Acquisition (OpenNeuro)
The script uses the `openneuro-py` library to communicate with the [OpenNeuro](https://openneuro.org/) platform. It targets the **Slackline study (ds003138)**, a high-quality diffusion MRI dataset.

To keep the demo fast, it only downloads:
- `participants.tsv` (Metadata)
- DWI sidecar files (`.json`) for subject `sub-82KK02101`.

This demonstrates that OptiConn is "BIDS-aware" and can pull data directly from public repositories.

### 2. Derivative Simulation
Tractography requires preprocessed data (FODs, tissue masks, parcellations). In a real-world study, you would run [QSIPrep](https://qsiprep.readthedocs.io/) and [QSIRecon](https://qsiprep.readthedocs.io/en/latest/reconstruction.html).

Since these steps take 5-10 hours per subject, the demo **mocks** the expected directory structure and files:
- **WM FOD**: `sub-82KK02101_ses-3_label-WM_dwimap.mif.gz`
- **ACT Mask**: `sub-82KK02101_ses-3_space-T1w_seg-hsvs_probseg.nii.gz`
- **Parcellation**: `sub-82KK02101_ses-3_seg-desikan_dseg.nii.gz`

This allows you to test the **optimization layer** immediately.

### 3. Bundle Discovery (`mrtrix-discover`)
OptiConn includes a discovery tool that scans your derivatives folder to find the files created in Step 2.

**Command executed by the demo:**
```bash
opticonn mrtrix-discover \
    --derivatives-dir demo_mrtrix/derivatives \
    --subject sub-82KK02101 \
    --session ses-3 \
    --atlas desikan \
    -o demo_mrtrix/mrtrix_bundle.json
```

This produces a `mrtrix_bundle.json` file. This file is the "source of truth" for the optimizer, ensuring that all downstream steps use the correct anatomical and diffusion data.

### 4. Bayesian Optimization (`tune-bayes`)
This is the heart of OptiConn. It uses Gaussian Process regression to find the tractography parameters that maximize network quality.

**Command executed by the demo:**
```bash
opticonn --backend mrtrix --dry-run tune-bayes \
    -i demo_mrtrix/mrtrix_bundle.json \
    -o demo_mrtrix/tuning_results \
    --subject sub-82KK02101 \
    --n-iterations 2
```

**What happens during this step:**
- **Parameter Selection**: The optimizer picks a set of parameters (e.g., `cutoff=0.17`, `angle=36`, `step=1.2`).
- **Tractography (Simulated)**: It generates the `tckgen` and `tck2connectome` commands.
- **Scoring**: In a real run, it would extract the connectivity matrix, compute graph metrics (Density, Efficiency, Small-Worldness), and calculate a `quality_score_raw`.
- **Learning**: The optimizer "learns" from the score and picks better parameters for the next iteration.

---

## Interpreting the Output

After running the demo, you will see the following in the `demo_mrtrix/` directory:
- `mrtrix_bundle.json`: The discovered paths for the subject.
- `derivatives/`: The mocked BIDS structure.
- `tuning_results/`: (If run without `--dry-run`) This would contain the optimization logs, CSV tables of all iterations, and the `bayesian_optimization_results.json` file.

## Next Steps

To run a real optimization on your own data:
1. Ensure MRtrix3 is installed.
2. Run `opticonn mrtrix-discover` on your QSIRecon derivatives.
3. Run `opticonn tune-bayes` without the `--dry-run` flag.
4. Use `opticonn select` to promote the best candidate.
5. Use `opticonn apply` to process your entire cohort.
