# OptiConn Pipeline

There is no gold standard for "correct" tractography parameters — no ground-truth
connectome to check a candidate atlas/threshold/tracking combination against.
**OptiConn** does not claim to find the optimal parameter set. Instead, it screens
candidates on explicit, testable criteria: it tracks each candidate multiple times per
subject and scores it by **repeat-run discriminability** — how well repeated runs of
the same subject can be told apart from other subjects, above tracking noise — and
rejects candidates whose connectomes are implausible (density/isolated-node gates).
The top-ranked, defensible setting is then applied to the full dataset.

```bash
# 1. Search candidate parameters (Bayesian search or grid/random sweep)
python opticonn.py tune-grid -i /data/pilot -o studies/demo_grid --quick

# 2. Promote the best atlas/metric within the discriminability-screened combo
python opticonn.py select -i studies/demo_grid/sweep-*/optimize

# 3. Apply the selected setting to the full dataset
python opticonn.py apply -i /data/all_subjects \
  --optimal-config studies/demo_grid/sweep-*/optimize/selected_candidate.json \
  -o studies/final_analysis
```

Selection happens at two levels, and they use different criteria:

1. **Combo (parameter-set) selection — by discriminability.** The grid sweep
   (`tune-grid`, on either backend) tracks every candidate parameter set repeatedly per
   subject and keeps the one with the highest repeat-run discriminability, after the
   density/isolated-node gates (`scripts/reliability.py`).
2. **Atlas/metric promotion — by quality score.** `opticonn select` then promotes the
   best atlas/connectivity-metric pair *within* that already-screened combo, using the
   composite QA score and wave consistency (`scripts/optimal_selection.py`).

The Bayesian sampler (`tune-bayes`) is a candidate *proposer*: on the DSI Studio backend
it explores by composite quality score, not by discriminability. To re-screen its
proposals by discriminability, hand them to the grid runner
(`python scripts/cross_validation_bootstrap_optimizer.py --candidates-from-bayes
<bayesian_optimization_results.json> ...`; not yet surfaced as an `opticonn` flag). On the
MRtrix3 backend, `tune-bayes` proposes by quality score but makes its *final* pick by
discriminability.

DSI Studio is the default tractography backend. An MRtrix3 backend is also available
for QSIRecon/QSIPrep users (`--backend mrtrix`, see below); both backends screen
`tune-grid` candidates the same way, by repeat-run discriminability.

## Third-party software (not redistributed)

This repository contains only OptiConn source code under the MIT License. It does **not** include or redistribute third-party executables or vendored third-party libraries.

OptiConn depends on third-party software that you must install separately, including:
- Python packages listed in `pyproject.toml` (installed via your Python environment)
- DSI Studio (installed separately; OptiConn calls your local DSI Studio executable)

---

## Docker (reproducible builds)

If you want to simulate a "fresh OS" installation for reproducibility (e.g., for JOSS review), you can build OptiConn in a clean Docker image. The build downloads Python packages during `docker build`.

Important: the image intentionally does **not** bundle or download DSI Studio. For full tractography runs you must provide a compatible DSI Studio binary yourself.

Build a minimal runtime image:

```bash
docker build --target runtime -t opticonn:runtime .
docker run --rm opticonn:runtime --help
```

On Apple Silicon (arm64), if you need to run an x86_64 (amd64) DSI Studio binary inside Docker, build and run with:

```bash
docker build --platform=linux/amd64 --target runtime -t opticonn:runtime-amd64 .
docker run --rm --platform=linux/amd64 opticonn:runtime-amd64 --help
```

Build and serve documentation:

```bash
docker build --target docs -t opticonn:docs .
docker run --rm -p 8000:8000 opticonn:docs
```

If you have a Linux DSI Studio binary available on the host, you can mount it and point `DSI_STUDIO_PATH` at it (example):

```bash
docker run --rm \
  --platform=linux/amd64 \
  -e DSI_STUDIO_PATH=/dsi/dsi_studio \
  -v /absolute/path/to/dsi_studio:/dsi/dsi_studio:ro \
  opticonn:runtime \
  python scripts/validate_setup.py --config configs/braingraph_default_config.json --no-input-test
```

Note: the mounted DSI Studio executable must match the container architecture (e.g. `linux/amd64`). Mounting a macOS `.app` binary into a Linux container will not work.

---

## 🔧 Installation Guide

### 1. Prerequisites

- Python 3.10 or newer
- Git and basic build tools (`build-essential` on Linux, Xcode Command Line Tools on macOS)
- [DSI Studio](https://dsi-studio.labsolver.org/download.html) installed locally (Required: OptiConn depends on DSI Studio for all tractography operations)
- At least 20 GB free disk space for intermediate results

Note: the repository does not ship DSI Studio or Python dependencies; the installer sets up a local environment on your machine.

> OptiConn is supported on macOS and Linux. On Windows, run it inside WSL2 with Ubuntu — see [docs/windows_wsl.md](docs/windows_wsl.md).

### 2. Quick install (macOS & Linux)

```bash
# Clone the repository
git clone https://github.com/MRI-Lab-Graz/opticonn.git
cd opticonn

# Provision the curated virtual environment with DSI Studio path
# Linux example:
bash install.sh --dsi-path /usr/local/bin/dsi_studio

# macOS example:
bash install.sh --dsi-path /Applications/dsi_studio.app/Contents/MacOS/dsi_studio

# Activate the virtual environment
source braingraph_pipeline/bin/activate
```

**Note:** The `--dsi-path` argument is required and must point to the DSI Studio executable. Use `bash install.sh --help` for more information.

### 3. Verify the setup

```bash
source braingraph_pipeline/bin/activate
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

The validator checks Python dependencies, DSI Studio accessibility (via the `DSI_STUDIO_PATH` environment variable set during installation), and configuration basics.

---

## 📂 Demo Data

For testing the pipeline, we recommend using our open dataset hosted on OpenNeuro:

**Dataset ds003138**: [https://openneuro.org/datasets/ds003138/versions/1.0.1](https://openneuro.org/datasets/ds003138/versions/1.0.1)

This dataset contains diffusion MRI data compatible with the pipeline and is ideal for running initial tests or demonstrations.

The lightweight CLI demo pulls a public HCP Young Adult sample from the data-hcp/lifespan GitHub release mirror ([100307.qsdr.fz](https://github.com/data-hcp/lifespan/releases/download/hcp-ya/100307.qsdr.fz)) to keep the download small.

---

## 🚀 The OptiConn Workflow

OptiConn offers two powerful methods for parameter discovery: **Bayesian Optimization** (`tune-bayes`, recommended for efficiency) and **Grid/Random Search** (`tune-grid`, for exhaustive baselines).

### Method A: tune-bayes (Recommended) ⭐

Proposes candidate parameters using Gaussian Processes, converging on a defensible candidate in 20-50 iterations (vs. thousands for grid search). On the DSI Studio backend it explores and ranks by composite quality score, *not* by discriminability — to screen its proposals by discriminability, feed them to the grid runner with `--candidates-from-bayes` (see the two-level selection note at the top of this README). With `--backend mrtrix`, the final pick among the proposed candidates is made by discriminability.

```bash
# Run Bayesian optimization with subject sampling
python opticonn.py tune-bayes \
  -i /data/fiber_bundles \
  -o studies/demo_bayes \
  --config configs/braingraph_default_config.json \
  --modalities qa fa \
  --n-iterations 30 \
  --sample-subjects
```

**Why use this?**
- **Fast:** Converges in 2-3 hours.
- **Smart:** Learns from previous iterations to find the "sweet spot".
- **Robust:** `--sample-subjects` ensures parameters work across the population, not just one subject.

**Output:**
**Outputs (per modality):**
- `bayesian_optimization_manifest.json`: index of modality-specific runs.
- `<output>/<modality>/bayesian_optimization_results.json`: top-ranked parameters for that modality.
- `<output>/<modality>/iterations/`: per-iteration logs and artifacts.

### Method B: tune-grid (Grid/Random)

Systematic cross-validation across two independent waves. Best for establishing a rigorous baseline or testing a specific, small set of combinations.

```bash
python opticonn.py tune-grid \
  -i /data/fiber_bundles \
  -o studies/demo_grid \
  --quick
```

**Key options:**
- `--quick`: Uses tiny micro tuning for fast demonstration.
- `--subjects N`: Number of subjects to use for validation (default: 3).

---

### Step 2: Select (`opticonn select`)

Analyze results from either method and select the top-ranked parameter combination:

```bash
# For Bayesian results:
python opticonn.py select \
  -i studies/demo_bayes --modality qa
  

# For Grid/Random tuning results:
python opticonn.py select \
  -i studies/demo_grid/sweep-<uuid>/optimize \
  
```

**What it does:**
- **Bayesian:** Displays the top-ranked parameters found and prepares the config for application.
- **Grid/Random:** Promotes the best atlas/connectivity-metric pair *within* the combo `tune-grid` already screened by discriminability, ranking those pairs by QA score and consistency across waves. `select` itself does not re-rank combos by discriminability.
- Optionally launches interactive web dashboard with `--interactive` (grid outputs only).

### Step 3: Apply to Full Dataset (`opticonn apply`)

Apply the selected parameters to your complete dataset:

```bash
python opticonn.py apply \
  -i /data/all_subjects \
  --optimal-config studies/demo_bayes/qa/bayesian_optimization_results.json \
  -o studies/final_analysis
```

**What it does:**
- Extracts connectivity using the selected parameters for all subjects
- Runs full optimization and selection pipeline
- Generates analysis-ready CSV files

**Final output:**
```text
studies/final_analysis/
├── 01_connectivity/        # Connectivity matrices per atlas
├── 02_optimization/        # Quality scores and rankings
└── 03_selection/          # Analysis-ready CSVs
    ├── FreeSurferSeg_qa_analysis_ready.csv
    ├── FreeSurferDKT_Cortical_qa_analysis_ready.csv
    └── optimal_selection_summary.txt
```

---

## ⚡ Quick Start Examples

### Recommended: tune-bayes Workflow

```bash
# 1. Propose candidate parameters (smart search)
python opticonn.py tune-bayes \
  -i /data/pilot \
  -o studies/bayes_opt \
  --config configs/braingraph_default_config.json \
  --modalities qa fa \
  --n-iterations 30 \
  --sample-subjects

# 2. Select results
python opticonn.py select \
  -i studies/bayes_opt --modality qa
  

# 3. Apply to full dataset
python opticonn.py apply \
  -i /data/full_dataset \
  --optimal-config studies/bayes_opt/qa/bayesian_optimization_results.json \
  -o studies/final
```

### Alternative: tune-grid Workflow

```bash
# 1. Run tune-grid
python opticonn.py tune-grid -i /data/pilot -o studies/test --quick

# 2. Select top-ranked candidate
python opticonn.py select -i studies/test/sweep-*/optimize

# 3. Apply to full dataset
python opticonn.py apply \
  -i /data/full_dataset \
  --optimal-config studies/test/sweep-*/optimize/selected_candidate.json \
  -o studies/final
```

---

## 🧪 CLI Demo (everything at a glance)

Run these on a tiny pilot subset to see all major commands and options:

```bash
# Bayesian optimization (recommended)
python opticonn.py tune-bayes \
  -i /data/pilot \
  -o demo/bayes \
  --config configs/braingraph_default_config.json \
  --n-iterations 15 \
  --sample-subjects

# Grid/random tuning (quick demo)
python opticonn.py tune-grid \
  -i /data/pilot \
  -o demo/grid \
  --quick \
  --subjects 2 \
  --max-parallel 2

# Select top-ranked candidate (works for both outputs)
python opticonn.py select -i demo/bayes --modality qa
python opticonn.py select -i demo/grid/sweep-*/optimize --prune-nonbest

# Apply to a larger dataset using the chosen config
python opticonn.py apply -i /data/all_subjects \
  --optimal-config demo/bayes/qa/bayesian_optimization_results.json \
  -o demo/final

# (Optional) Run the classic pipeline in one shot
python opticonn.py pipeline --step all \
  --input /data/all_subjects \
  --output demo/pipeline \
  --config configs/braingraph_default_config.json \
  

# (Optional) Sensitivity analysis for interpretability
python opticonn.py sensitivity -i /data/pilot -o demo/sensitivity \
  --config configs/braingraph_default_config.json \
  --parameters fa_threshold turning_angle tract_count \

### New: one-shot demo helper

For a minimal end-to-end run (download sample data, tune-bayes, select, apply):

```bash
python scripts/opticonn_demo.py --step all

# Run individual phases if desired
python scripts/opticonn_demo.py --step 1   # tune-bayes
python scripts/opticonn_demo.py --step 2   # select
python scripts/opticonn_demo.py --step 3   # apply
```

### Cross-validation demo (seeding from Bayesian)

If you want to validate parameter robustness across waves using the same tiny sample:

```bash
python scripts/opticonn_cv_demo.py --workspace demo_workspace_cv

# Optional: run specific modalities
python scripts/opticonn_cv_demo.py --workspace demo_workspace_cv --modalities qa fa
```

By default it seeds **per modality** from `demo_workspace/results/bayes/<modality>/bayesian_optimization_results.json` (override with `--from-bayes`), fixes the metrics/atlases from your base config, and runs two bootstrap waves with 3 subjects each.
```

---

## 🔧 Advanced: Direct Pipeline Execution

For users who already know which parameters they want to use, the `pipeline` command runs the traditional extraction → optimization → selection workflow:

```bash
python opticonn.py pipeline --step all \
  --input /data/fiber_bundles \
  --output studies/direct_run \
  --config configs/braingraph_default_config.json
```

**Pipeline steps:**

| Step | Purpose | Output |
|------|---------|--------|
| 01 | Connectivity extraction | `01_connectivity/` with per-atlas matrices |
| 02 | Network quality optimization | `02_optimization/optimized_metrics.csv` |
| 03 | Quality-based selection | `03_selection/*_analysis_ready.csv` |

**Step control:**
- `--step 01`: Run only extraction
- `--step 02`: Run only optimization (requires existing 01 output)
- `--step 03`: Run only selection (requires existing 02 output)
- `--step analysis`: Run steps 02+03 (skip extraction)
- `--step all`: Run complete pipeline 01→02→03

---

## 🎯 Deep Dive: Bayesian Optimization

Bayesian search provides an efficient alternative to grid/random search for proposing candidate tractography parameters. Instead of exhaustively testing all combinations, it uses a Gaussian Process to model the parameter-quality relationship and strategically samples the most promising regions. Note that this sampler's own objective is the composite quality score; discriminability screening of its proposals is a separate step (`--candidates-from-bayes` on the grid runner, or `--backend mrtrix`, whose final pick is discriminability-based).

### Subject Sampling Strategies

The optimizer supports three strategies for handling subject variability:

| Strategy | Flag | Behavior | Runtime | Robustness | Use Case |
|----------|------|----------|---------|------------|----------|
| **Single Subject** | (default) | Same subject for all iterations | ~2 hours | Low | Quick exploration |
| **Bootstrap** | `--n-bootstrap 3` | Same 3 subjects for all iterations | ~6 hours | Medium | Stable optimization |
| **Subject Sampling** ⭐ | `--sample-subjects` | Different subject per iteration | ~2 hours | **High** | **Production (recommended)** |

**Why Subject Sampling works:** The Gaussian Process models `f(params) = signal + noise`, where noise represents subject variability. By seeing different subjects, it learns parameters that consistently perform well across the population.

### Configuration File

The optimizer uses `sweep_parameters` in your config JSON to define parameter ranges:

```json
{
  "sweep_parameters": {
    "description": "Bayesian optimization parameter ranges",
    "tract_count_range": [10000, 200000],
    "fa_threshold_range": [0.05, 0.3],
    "min_length_range": [5, 50],
    "turning_angle_range": [30.0, 90.0],
    "step_size_range": [0.5, 2.0],
    "track_voxel_ratio_range": [1.0, 5.0],
    "connectivity_threshold_range": [0.0001, 0.01]
  }
}
```

All ranges are `[min, max]` bounds that the Bayesian optimizer will intelligently sample.

---

## 🧠 OptiConn CLI Commands Reference

### Global Options (all commands)

- `--version`: Show OptiConn version
- `--dry-run`: Print commands without executing them

### `tune-bayes` - Bayesian optimization

```bash
python opticonn.py tune-bayes -i DATA_DIR -o OUTPUT_DIR --config CONFIG [options]
```

**Required:**
- `-i, --data-dir`: Directory containing .fz or .fib.gz files
- `-o, --output-dir`: Output directory for optimization results
- `--config`: Base configuration JSON file

**Optional:**
- `--n-iterations N`: Number of optimization iterations (default: 30)
- `--n-bootstrap N`: Bootstrap samples per evaluation (default: 3)
- `--max-workers N`: Parallel workers (default: 1)
- `--sample-subjects`: Use different subject per iteration (recommended)
- `--verbose`: Show detailed progress

### `tune-grid` - Grid/Random tuning

```bash
python opticonn.py tune-grid -i DATA_DIR -o OUTPUT_DIR [options]
```

**Required:**
- `-i, --data-dir`: Directory containing .fz or .fib.gz files
- `-o, --output-dir`: Output directory for tuning results

**Optional:**
- `--quick`: Run tiny demonstration tuning (configs/sweep_micro.json)
- `--subjects N`: Number of subjects for validation (default: 3)
- `--max-parallel N`: Max combinations to run in parallel per wave
- `--extraction-config`: Override extraction config
- `--no-report`: Skip quality and Pareto reports
- `--no-validation`: Skip setup validation
- `--verbose`: Show DSI Studio commands and detailed progress

### `select` - Promote best candidate

```bash
python opticonn.py select -i INPUT_PATH [options]
```

**Required:**
- `-i, --input-path`: Grid-tuning optimize directory **or** Bayesian results JSON file

**Optional:**
- `--prune-nonbest`: Delete non-optimal combo outputs to save disk space

### `apply` - Apply optimal parameters to full dataset

```bash
python opticonn.py apply -i DATA_DIR --optimal-config CONFIG [-o OUTPUT_DIR] [options]
```

**Required:**
- `-i, --data-dir`: Directory containing full dataset (.fz or .fib.gz files)
- `--optimal-config`: Path to selected_candidate.json from selection step

**Optional:**
- `-o, --output-dir`: Output directory (default: analysis_results)
- `--analysis-only`: Run only analysis on existing extraction outputs
- `--candidate-index N`: Select specific candidate by 1-based index (default: 1)
- `--verbose`: Show detailed progress
- `--quiet`: Minimal console output

### `pipeline` - Advanced pipeline execution

```bash
python opticonn.py pipeline --step STEP [options]
```

**Options:**
- `--step {01,02,03,all,analysis}`: Which pipeline step(s) to run
- `-i, --input`: Input directory or file
- `-o, --output`: Output directory
- `--config`: Configuration file (default: configs/braingraph_default_config.json)
- `--data-dir`: Alternative way to specify input data
- `--cross-validated-config`: Use cross-validation outputs
- `--quiet`: Minimal console output

---

## 🧬 MRtrix3 backend (opt-in)

OptiConn's default tractography backend is DSI Studio. An alternative MRtrix3 backend is
available for QSIRecon/QSIPrep users: it re-runs `tckgen`/`tcksift2`/`tck2connectome` on
already-preprocessed QSIRecon derivatives instead of DSI Studio's `.fz`/`.fib.gz` pipeline, and
scores candidate parameters by repeat-run discriminability/repeatability (see
`scripts/reliability.py`) rather than a single QA pass.

Opt in with `--backend mrtrix` on any `opticonn` command that supports it (currently `tune-grid`,
`tune-bayes`, and `apply`):

```bash
# Fixed one-subject file bundle: pass the config as -i (a FILE path selects config mode)
python -m scripts.opticonn_hub --backend mrtrix tune-grid \
  -i configs/mrtrix_default_sweep.json \
  -o /path/to/opticonn/mrtrix_out \
  --subject sub-01

# Auto-discovery across several subjects (a DIRECTORY for -i selects discovery mode);
# --atlas is required here, and >=2 subjects is what makes discriminability computable
python -m scripts.opticonn_hub --backend mrtrix tune-grid \
  -i /path/to/derivatives \
  -o /path/to/opticonn/mrtrix_out \
  --subject sub-01 sub-02 sub-03 \
  --atlas Schaefer200
```

Add `--dry-run` (before the subcommand) to print the MRtrix commands without running them.

Input expectations:
- A QSIRecon derivatives directory (WM FOD, ACT tissue image, atlas parcellation + labels), passed
  either as a **file** path to `-i` (a fixed one-subject `mrtrix_tune` config bundle) or as a
  **directory** path to `-i` (auto-discovery across one or more `--subject` values, needed for
  cross-subject discriminability). The hub decides between the two by whether `-i` is a file
  or a directory.
- `--atlas` selects which parcellation to use; it is **required** in discovery mode and
  optional in config mode (where it picks among several configured parcellations).

`configs/mrtrix_default_sweep.json` is a starter config (edit `inputs.bundle` to point at your own
files, or generate one automatically — see below). It includes the same
`"reliability": {"repeats": 2, "density_range": [...], "max_isolated_fraction": ...}` block shape
used by the DSI Studio sweep, so repeat-run reliability gating is consistent across backends.

For the full option reference and worked examples, see:
- [`scripts/mrtrix_discover_bundle.README.md`](scripts/mrtrix_discover_bundle.README.md) — auto-discovering a bundle from QSIRecon outputs
- [`scripts/mrtrix_tune.README.md`](scripts/mrtrix_tune.README.md) — the `sweep`/`bayes`/`apply` MRtrix tuner itself

---

## 📌 Configuration Files

### `configs/braingraph_default_config.json`

Primary extraction configuration used by default in pipeline and tuning commands.

**Key settings:**
- `dsi_studio_cmd`: path to the DSI Studio executable
- `atlases`: which atlases to extract (e.g., FreeSurferDKT_Cortical, FreeSurferDKT_Tissue, FreeSurferSeg)
- `connectivity_values`: metrics such as `count`, `fa`, `qa`, `ncount2`
- `tract_count`, `thread_count`, and detailed `tracking_parameters`
- `connectivity_options`: output types and thresholds
- `sweep_parameters`: ranges (supports MATLAB-style strings like `0.3:0.2:0.7`) and sampling method (`grid`, `random`, `lhs`)

**Usage:**

```bash
# Validate configuration
python scripts/json_validator.py configs/braingraph_default_config.json

# Use in pipeline
python opticonn.py pipeline --step all \
   --input /path/to/fz \
   --output studies/custom_run \
   --config configs/my_custom_config.json
```

Schema reference: `dsi_studio_config_schema.json`

---

## 🧪 Complete Workflow Example

Below is a concrete session for a dataset stored in `/data/P124`:

1. **Activate the environment and set DSI Studio path**

   ```bash
   source braingraph_pipeline/bin/activate
   export DSI_STUDIO_CMD=/Applications/dsi_studio.app/Contents/MacOS/dsi_studio
   ```

2. **Run tune-grid on pilot data**

   ```bash
   python opticonn.py tune-grid \
     -i /data/P124/pilot_subjects \
     -o studies/p124_sweep \
    --subjects 3
   ```

3. **Select top-ranked parameters**

   ```bash
   python opticonn.py select \
    -i studies/p124_sweep/sweep-*/optimize
   ```

4. **Apply to full dataset**

   ```bash
   python opticonn.py apply \
     -i /data/P124/all_subjects \
    --optimal-config studies/p124_sweep/sweep-*/optimize/selected_candidate.json \
     -o studies/p124_final
   ```

5. **Review results**
   - `studies/p124_final/03_selection/*_analysis_ready.csv`: Ready for statistical analysis
   - `studies/p124_final/02_optimization/optimized_metrics.csv`: Quality scores
   - `studies/p124_final/03_selection/optimal_selection_summary.txt`: Summary report

---

## 🧠 Expert Settings & Advanced Toolkit

### Individual Script Control

While `opticonn` commands orchestrate the complete workflow, you can also run individual scripts directly for fine-grained control:

**Extraction (Step 01):**
```bash
python scripts/extract_connectivity_matrices.py \
  --config configs/braingraph_default_config.json \
  --batch \
  --input /data/fibers \
  --output studies/manual_run
```

**Optimization (Step 02):**
```bash
python scripts/metric_optimizer.py \
  --input studies/manual_run/01_connectivity \
  --output studies/manual_run/02_optimization
```

**Selection (Step 03):**
```bash
python scripts/optimal_selection.py \
  --input studies/manual_run/02_optimization \
  --output studies/manual_run/03_selection \
  --plots
```

### Utility Scripts

| Script | Purpose |
| ------ | ------- |
| `scripts/aggregate_network_measures.py` | Merge per-subject network metrics |
| `scripts/cross_validation_bootstrap_optimizer.py` | Multi-wave QA campaigns |
| `scripts/bootstrap_qa_validator.py` | Validate QA batches |
| `scripts/json_validator.py` | Validate configuration files |
| `scripts/quick_quality_check.py` | Spot-check diversity and sparsity |
| `scripts/pareto_view.py` | Generate Pareto fronts from diagnostics |
| `scripts/validate_setup.py` | Pre-flight environment check |

---

## 🗺️ Script Architecture

```mermaid
graph TD
    A[opticonn.py] --> B[scripts/opticonn_hub.py]
  B -->|tune-grid| H[scripts/cross_validation_bootstrap_optimizer.py]
  B -->|tune-bayes| K[scripts/bayesian_optimizer.py]
  B -->|select| I[scripts/optimal_selection.py]
    B -->|apply| C[scripts/run_pipeline.py]
    B -->|pipeline| C
    C --> D[scripts/extract_connectivity_matrices.py]
    C --> E[scripts/aggregate_network_measures.py]
    C --> F[scripts/metric_optimizer.py]
    C --> G[scripts/optimal_selection.py]
    H --> C
    H --> J[scripts/pareto_view.py]
```

**Key components:**
- `opticonn.py`: CLI entry point with venv bootstrapping
- `scripts/opticonn_hub.py`: Command router (`tune-bayes`, `tune-grid`, `select`, `apply`, `pipeline`)
- `scripts/run_pipeline.py`: Orchestrates 3-step workflow
- `scripts/extract_connectivity_matrices.py`: DSI Studio interface
- `scripts/cross_validation_bootstrap_optimizer.py`: Grid/random tuning engine
- `scripts/optimal_selection.py`: Candidate selection and final output generation

---

## 📊 Diagnostics & Pareto Analysis

Every grid-tuning combination writes `diagnostics.json` with parameters, scores, and network measures:

```text
studies/<name>/sweep-<uuid>/optimize/<wave>/combos/sweep_0001/diagnostics.json
```

### Generate Pareto Front

Surface combinations balancing quality, cost, and network properties:

```bash
python scripts/pareto_view.py \
  studies/sweep/optimize/bootstrap_qa_wave_1 \
  studies/sweep/optimize/bootstrap_qa_wave_2 \
  -o studies/sweep/optimize/optimization_results \
  --plot
```

**Outputs:**
- `pareto_front.csv`: Pareto-efficient combinations
- `pareto_candidates_with_objectives.csv`: All combos with objectives
- `pareto_front.png`: Visualization (with `--plot`)

**Tuning:**
- `--score selection_score`: Use Step 02 selection score
- `--density-range 0.08 0.25`: Adjust preferred density corridor

---

## ✅ Troubleshooting

### DSI Studio Not Found

Set the path explicitly:

```bash
export DSI_STUDIO_CMD=/path/to/dsi_studio  # macOS/Linux
```

Or update `configs/braingraph_default_config.json`:

```json
{
  "dsi_studio_cmd": "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio"
}
```

### Validate Configuration

Before long runs:

```bash
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

### Check Dependencies

```bash
source braingraph_pipeline/bin/activate
pip check
```

---

## 📚 Further Reading

- **DSI Studio Documentation**: https://dsi-studio.labsolver.org/
- **Configuration Schema**: `dsi_studio_config_schema.json`
- **Example Configs**: `configs/` directory
- **Script Help**: Add `--help` to any command for detailed options

---

## 🤝 Contributing

Issues and pull requests welcome at https://github.com/MRI-Lab-Graz/opticonn

---

## 📄 License

See `LICENSE` file for details.

---

## 📖 Citation

If you use OptiConn in your research, please cite:

```
[Citation details to be added - see CITATION.cff]
```
