# OptiConn Pipeline

**OptiConn** is an unbiased, modality-agnostic connectomics optimization and analysis toolkit. It automates the discovery of optimal tractography parameters through systematic cross-validation, then applies those parameters to generate analysis-ready brain connectivity datasets.

---

## 🔧 Installation Guide

### 1. Prerequisites

- Python 3.10 or newer (the bundled virtual environment targets 3.10)
- Git and basic build tools (`build-essential` on Linux, Xcode Command Line Tools on macOS)
- [DSI Studio](https://dsi-studio.labsolver.org/download.html) installed locally (Required: OptiConn depends on DSI Studio for all tractography operations)
- At least 20 GB free disk space for intermediate results

### 2. Quick install (macOS & Linux)

```bash
# Clone the repository
git clone https://github.com/MRI-Lab-Graz/braingraph-pipeline.git
cd braingraph-pipeline

# Provision the curated virtual environment with DSI Studio path
# Linux example:
bash install.sh --dsi-path /usr/local/bin/dsi_studio

# macOS example:
bash install.sh --dsi-path /Applications/dsi_studio.app/Contents/MacOS/dsi_studio

# Activate the virtual environment
source braingraph_pipeline/bin/activate
```

**Note:** The `--dsi-path` argument is required and must point to the DSI Studio executable. Use `bash install.sh --help` for more information.

### 3. Quick install (Windows PowerShell)

```powershell
git clone https://github.com/MRI-Lab-Graz/braingraph-pipeline.git
cd braingraph-pipeline

# Create the Windows venv with pinned dependencies
./install_windows.bat

# Activate and set the DSI Studio path
braingraph_pipeline\Scripts\activate.bat
setx DSI_STUDIO_CMD "C:\\Program Files\\dsi_studio\\dsi_studio.exe"
```

### 4. Verify the setup

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

---

## 🚀 The OptiConn Workflow

OptiConn offers two powerful methods for parameter discovery: **Bayesian Optimization** (recommended for efficiency) and **Grid Search** (for exhaustive baselines).

### Method A: Bayesian Optimization (Recommended) ⭐

Intelligently discovers optimal parameters using Gaussian Processes. Finds the best configuration in 20-50 iterations (vs. thousands for grid search).

```bash
# Run Bayesian optimization with subject sampling
python opticonn.py bayesian \
  -i /data/fiber_bundles \
  -o studies/demo_bayes \
  --config configs/braingraph_default_config.json \
  --n-iterations 30 \
  --sample-subjects \
  --no-emoji
```

**Why use this?**
- **Fast:** Converges in 2-3 hours.
- **Smart:** Learns from previous iterations to find the "sweet spot".
- **Robust:** `--sample-subjects` ensures parameters work across the population, not just one subject.

**Output:**
- `bayesian_optimization_results.json`: The best parameters found.
- `iterations/`: Detailed logs and results for every step.

### Method B: Parameter Sweep (Grid Search)

Systematic cross-validation across two independent waves. Best for establishing a rigorous baseline or testing a specific, small set of combinations.

```bash
python opticonn.py sweep \
  -i /data/fiber_bundles \
  -o studies/demo_sweep \
  --quick \
  --no-emoji
```

**Key options:**
- `--quick`: Uses tiny micro sweep for fast demonstration.
- `--subjects N`: Number of subjects to use for validation (default: 3).

---

### Step 2: Review & Select (`opticonn review`)

Analyze results from either method and select the best parameter combination:

```bash
# For Bayesian results:
python opticonn.py review \
  -i studies/demo_bayes/bayesian_optimization_results.json \
  --no-emoji

# For Sweep results:
python opticonn.py review \
  -o studies/demo_sweep/sweep-<uuid>/optimize \
  --no-emoji
```

**What it does:**
- **Bayesian:** Displays the best parameters found and prepares the config for application.
- **Sweep:** Automatically ranks candidates by QA scores and consistency across waves.
- Optionally launches interactive web dashboard with `--interactive` (Sweep only).

### Step 3: Apply to Full Dataset (`opticonn apply`)

Apply the optimal parameters to your complete dataset:

```bash
python opticonn.py apply \
  -i /data/all_subjects \
  --optimal-config studies/demo_bayes/bayesian_optimization_results.json \
  -o studies/final_analysis \
  --no-emoji
```

**What it does:**
- Extracts connectivity using optimal parameters for all subjects
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

### Recommended: Bayesian Workflow

```bash
# 1. Find optimal parameters (smart search)
python opticonn.py bayesian \
  -i /data/pilot \
  -o studies/bayes_opt \
  --config configs/braingraph_default_config.json \
  --n-iterations 30 \
  --sample-subjects \
  --no-emoji

# 2. Review results
python opticonn.py review \
  -i studies/bayes_opt/bayesian_optimization_results.json \
  --no-emoji

# 3. Apply to full dataset
python opticonn.py apply \
  -i /data/full_dataset \
  --optimal-config studies/bayes_opt/bayesian_optimization_results.json \
  -o studies/final \
  --no-emoji
```

### Alternative: Grid Search Workflow

```bash
# 1. Run parameter sweep
python opticonn.py sweep -i /data/pilot -o studies/test --quick --no-emoji

# 2. Auto-select best candidate
python opticonn.py review -o studies/test/sweep-*/optimize --no-emoji

# 3. Apply to full dataset
python opticonn.py apply \
  -i /data/full_dataset \
  --optimal-config studies/test/sweep-*/optimize/selected_candidate.json \
  -o studies/final --no-emoji
```

---

## 🔧 Advanced: Direct Pipeline Execution

For users who already know their optimal parameters, the `pipeline` command runs the traditional extraction → optimization → selection workflow:

```bash
python opticonn.py pipeline --step all \
  --input /data/fiber_bundles \
  --output studies/direct_run \
  --config configs/braingraph_default_config.json \
  --no-emoji
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

Bayesian optimization provides an intelligent alternative to grid/random search for finding optimal tractography parameters. Instead of exhaustively testing all combinations, it uses a Gaussian Process to model the parameter-quality relationship and strategically samples the most promising regions.

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

- `--no-emoji`: Disable emoji in console output (Windows-safe, recommended)
- `--version`: Show OptiConn version
- `--dry-run`: Print commands without executing them

### `sweep` - Run parameter sweep

```bash
python opticonn.py sweep -i DATA_DIR -o OUTPUT_DIR [options]
```

**Required:**
- `-i, --data-dir`: Directory containing .fz or .fib.gz files
- `-o, --output-dir`: Output directory for sweep results

**Optional:**
- `--quick`: Run tiny demonstration sweep (configs/sweep_micro.json)
- `--subjects N`: Number of subjects for validation (default: 3)
- `--max-parallel N`: Max combinations to run in parallel per wave
- `--extraction-config`: Override extraction config
- `--no-report`: Skip quality and Pareto reports
- `--no-validation`: Skip setup validation
- `--verbose`: Show DSI Studio commands and detailed progress

### `review` - Review and select best candidate

```bash
python opticonn.py review -o OUTPUT_DIR [options]
```

**Required:**
- `-o, --output-dir`: Sweep output directory (path to optimize folder)

**Optional:**
- `--interactive`: Launch interactive web dashboard for manual selection
- `--port N`: Port for Dash app (default: 8050)
- `--prune-nonbest`: Delete non-optimal combo outputs to save disk space

### `apply` - Apply optimal parameters to full dataset

```bash
python opticonn.py apply -i DATA_DIR --optimal-config CONFIG [-o OUTPUT_DIR] [options]
```

**Required:**
- `-i, --data-dir`: Directory containing full dataset (.fz or .fib.gz files)
- `--optimal-config`: Path to selected_candidate.json from review step

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

## 📌 Configuration Files

### `configs/braingraph_default_config.json`

Primary extraction configuration used by default in pipeline and sweep commands.

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

2. **Run parameter sweep on pilot data**

   ```bash
   python opticonn.py sweep \
     -i /data/P124/pilot_subjects \
     -o studies/p124_sweep \
     --subjects 3 \
     --no-emoji
   ```

3. **Review and select best parameters**

   ```bash
   # Automatic selection
   python opticonn.py review \
     -o studies/p124_sweep/sweep-*/optimize \
     --no-emoji
   
   # Or use interactive dashboard
   python opticonn.py review \
     -o studies/p124_sweep/sweep-*/optimize \
     --interactive \
     --no-emoji
   ```

4. **Apply to full dataset**

   ```bash
   python opticonn.py apply \
     -i /data/P124/all_subjects \
     --optimal-config studies/p124_sweep/sweep-*/optimize/selected_candidate.json \
     -o studies/p124_final \
     --no-emoji
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
  --output studies/manual_run \
  --no-emoji
```

**Optimization (Step 02):**
```bash
python scripts/metric_optimizer.py \
  --input studies/manual_run/01_connectivity \
  --output studies/manual_run/02_optimization \
  --no-emoji
```

**Selection (Step 03):**
```bash
python scripts/optimal_selection.py \
  --input studies/manual_run/02_optimization \
  --output studies/manual_run/03_selection \
  --plots \
  --no-emoji
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
    B -->|sweep| H[scripts/cross_validation_bootstrap_optimizer.py]
    B -->|review| I[scripts/optimal_selection.py]
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
- `scripts/opticonn_hub.py`: Command router (`sweep`, `review`, `apply`, `pipeline`)
- `scripts/run_pipeline.py`: Orchestrates 3-step workflow
- `scripts/extract_connectivity_matrices.py`: DSI Studio interface
- `scripts/cross_validation_bootstrap_optimizer.py`: Parameter sweep engine
- `scripts/optimal_selection.py`: Candidate selection and final output generation

---

## 📊 Diagnostics & Pareto Analysis

Every sweep combination writes `diagnostics.json` with parameters, scores, and network measures:

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
  --plot --no-emoji
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

### Windows Unicode Issues

Always use `--no-emoji` on Windows to avoid console encoding errors:

```bash
python opticonn.py sweep -i /data -o studies/run1 --no-emoji
```

### DSI Studio Not Found

Set the path explicitly:

```bash
export DSI_STUDIO_CMD=/path/to/dsi_studio  # macOS/Linux
setx DSI_STUDIO_CMD "C:\path\to\dsi_studio.exe"  # Windows
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

Issues and pull requests welcome at https://github.com/MRI-Lab-Graz/braingraph-pipeline

---

## 📄 License

See `LICENSE` file for details.

---

## 📖 Citation

If you use OptiConn in your research, please cite:

```
[Citation details to be added - see CITATION.cff]
```
