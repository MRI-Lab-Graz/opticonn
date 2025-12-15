# OptiConn Steps & CLI Reference

## What is OptiConn?

**OptiConn** is an unbiased, modality-agnostic connectomics optimization and analysis toolkit.
It automates (1) discovery of robust tractography parameters via systematic evaluation (Bayesian or cross-validated grid/random), then (2) applies those parameters to generate analysis-ready brain connectivity datasets.

OptiConn is built around **DSI Studio** for tractography and connectivity extraction.

---

## Installation (macOS / Linux only)

> Windows is **not supported** in this release.

### Prerequisites

- Python 3.10+
- Git + build tools (Xcode CLT on macOS, `build-essential` on Linux)
- DSI Studio installed locally (required)

### Install (curated virtual environment)

```bash
git clone https://github.com/MRI-Lab-Graz/opticonn.git
cd opticonn

# macOS example
bash install.sh --dsi-path /Applications/dsi_studio.app/Contents/MacOS/dsi_studio

# Linux example
# bash install.sh --dsi-path /usr/local/bin/dsi_studio

source braingraph_pipeline/bin/activate
```

What the installer does (high-level):

- Creates a virtual environment at `braingraph_pipeline/` using `uv`
- Installs OptiConn in editable mode with extras
- Persists `DSI_STUDIO_PATH` into the venv activation script so all pipeline steps can find DSI Studio

### Verify the setup

```bash
source braingraph_pipeline/bin/activate
python scripts/validate_setup.py --config configs/          braingraph_default_config.json
```

---

## Running OptiConn

OptiConn is exposed through the repository entrypoint:

```bash
python opticonn.py --help
python opticonn.py <command> --help
```

### Global options

These flags are accepted by the main CLI parser:

- `--version`: print the OptiConn version
- `--dry-run`: **currently defined but not executed by the hub** (no-op at the moment)

### Environment variables

- `DSI_STUDIO_PATH`: set by `install.sh`; used to resolve `dsi_studio_cmd` when configs use the generic `dsi_studio`
- `OPTICONN_SKIP_VENV=1`: disables the venv auto-bootstrap in `opticonn.py` (advanced; normally you want the curated venv)

---

## The standard 3-step OptiConn workflow

Most users follow:

1. **Step 1**: `tune-bayes` (recommended) or `tune-grid`
2. **Step 2**: `select`
3. **Step 3**: `apply`

The sections below document every `opticonn` command and flag.

---

## Step 1A (recommended): `tune-bayes`

Bayesian optimization searches tractography parameters efficiently (typically 20–50 evaluations instead of large grids).

### Command

```bash
python opticonn.py tune-bayes -i <pilot_data_dir> -o <output_dir> --config <base_config.json> --modalities <qa|fa|...>
```

### Required arguments

- `-i, --data-dir`: directory containing `.fz` or `.fib.gz` files
- `-o, --output-dir`: output directory for optimization results
- `--config`: base configuration JSON file

### Options

- `--n-iterations <int>`: number of Bayesian iterations (default: `30`)
- `--n-bootstrap <int>`: number of bootstrap samples per evaluation (default: `3`)
- `--max-workers <int>`: parallel workers (default: `1` = sequential)
- `--modalities <m1> [m2 ...]`: one or more modalities to optimize **separately** (e.g. `qa fa`). Here `qa` refers to **Quantitative Anisotropy** (not “quality assurance”). If omitted, OptiConn infers a single default modality from the config (prefers `qa` if present).
- `--sample-subjects`: sample a different subject each iteration (recommended for quick pilot runs)
- `--verbose`: more detailed optimization logging

### Outputs

- `<output_dir>/bayesian_optimization_manifest.json` (index of modality-specific runs)
- For each selected modality `<m>`:
  - `<output_dir>/<m>/bayesian_optimization_results.json`
  - `<output_dir>/<m>/iterations/` (logs and per-iteration artifacts)

### Demo mapping

- Run just Step 1: `python scripts/opticonn_demo.py --step 1 --n-iterations 4`
- Full demo workflow (Steps 1–3): `python scripts/opticonn_demo.py --step all`
- See also: [Demos](demos.md#bayesian-apply-quick-demo)

---

## Step 1B (cross-validated baseline): `tune-grid`

Grid/random tuning runs a **two-wave cross-validation bootstrap optimizer** to evaluate parameter combinations more exhaustively.

### Command

```bash
python opticonn.py tune-grid -i <pilot_data_dir> -o <output_dir>
```

### Required arguments

- `-i, --data-dir`: directory containing `.fz` or `.fib.gz` files for tuning
- `-o, --output-dir`: base output directory (a `sweep-<uuid>/` folder is created under it)

### Options

- `--quick`: tiny demo sweep (uses `configs/sweep_micro.json`)
- `--subjects <int>`: subjects per wave (default: `3`)
- `--max-parallel <int>`: max combinations to run in parallel per wave
- `--extraction-config <path>`: override extraction config used in auto-generated waves
- `--config <path>`: optional config; can be either an extraction-like config or a master optimizer config
- `--no-validation`: skip `scripts/validate_setup.py` before running
- `--no-report`: skip quick quality and Pareto reports after tuning
- `--verbose`: show DSI Studio commands and detailed progress per combination
- `--auto-select`: **deprecated** legacy behavior (selection now happens via `opticonn select`)

### Outputs

- `<output_dir>/sweep-<uuid>/optimize/` (wave outputs + logs)

### Demo mapping

- Quick grid demo: `python opticonn.py tune-grid -i demo_workspace_cv/data -o demo_workspace_cv/results/grid --quick`
- Cross-validation demo driver: `python scripts/opticonn_cv_demo.py --workspace demo_workspace_cv`
- See also: [Demos](demos.md#cross-validation-demo-seeded-from-bayes)

---

## Step 2: `select`

Selects the best candidate from either:

- `bayesian_optimization_results.json` for a specific modality (Bayesian; e.g. `<output>/<modality>/bayesian_optimization_results.json`)
- An `optimize/` directory produced by cross-validated tuning (grid/random)

### Command

```bash
python opticonn.py select -i <path>
```

If your Step 1 Bayesian tuning was run per-modality (via `--modalities`), you can also point `select` at the base output directory and pick a modality:

```bash
python opticonn.py select -i <bayes_output_dir> --modality <qa|fa|...>
```

### Required arguments

- `-i, --input-path`: path to Bayesian results JSON **or** an optimize directory
  - For per-modality Bayesian tuning you can pass the base output directory and add `--modality <qa|fa|...>`.

### Options

- `--prune-nonbest`: for grid outputs, delete non-optimal combo results after selection to save disk space

### Outputs

- For grid outputs: `<optimize_dir>/selected_candidate.json`
- For Bayesian outputs: prints the best parameters and a suggested next `apply` command

### Demo mapping

- Run just Step 2 (expects Step 1 outputs in the demo workspace): `python scripts/opticonn_demo.py --step 2`

---

## Step 3: `apply`

Applies optimal parameters to a dataset and runs the full analysis pipeline (connectivity extraction + optimization + selection).

### Command

```bash
python opticonn.py apply -i <full_data_dir> --optimal-config <selected_candidate.json|bayes_results.json> -o <output_dir>
```

### Required arguments

- `-i, --data-dir`: directory containing the full dataset (`.fz` or `.fib.gz`)
- `--optimal-config`: either:
  - `selected_candidate.json` produced by `opticonn select` on grid outputs, or
  - `bayesian_optimization_results.json` produced by `tune-bayes`

### Options

- `-o, --output-dir`: output directory (default: `analysis_results`)
- `--analysis-only`: skip connectivity extraction and re-run analysis on existing outputs
- `--candidate-index <int>`: if the optimal config contains multiple candidates, choose by 1-based index (default: `1`)
- `--verbose`: show detailed progress and DSI Studio commands
- `--quiet`: reduce console output
- `--skip-extraction`: deprecated alias for `--analysis-only`

### Outputs

- `<output_dir>/selected/01_connectivity/`
- `<output_dir>/selected/02_optimization/`
- `<output_dir>/selected/03_selection/`

### Demo mapping

- Run just Step 3 (expects Step 1 outputs in the demo workspace): `python scripts/opticonn_demo.py --step 3`
- Full demo workflow (Steps 1–3): `python scripts/opticonn_demo.py --step all`

---

## Cross-validation (advanced)

If you want cross-validation with **Bayesian seeding** (not currently surfaced by `opticonn tune-grid`), run the underlying optimizer directly.

### Script: `cross_validation_bootstrap_optimizer.py`

```bash
python scripts/cross_validation_bootstrap_optimizer.py \
  -i <pilot_data_dir> \
  -o <output_dir> \
  --extraction-config <extraction_config.json> \
  --from-bayes <bayesian_optimization_results.json> \
  --subjects 3 \
  --max-parallel 1 \
  --verbose
```

Notable flags:

- `--from-bayes <path>`: seed best parameters from Bayesian results
- `--candidates-from-bayes <path>`: evaluate top-K Bayes candidates (no sweep sampling)
- `--bayes-top-k <int>`: number of Bayes candidates to evaluate (default: `3`)
- `--random-baseline-k <int>`: optionally add K random candidates from `sweep_parameters` for comparison
- `--single-wave`: run one comprehensive wave instead of 2-wave cross-validation
- `--dry-run`: generate configs and summarize actions without executing

### Demo mapping

- `python scripts/opticonn_cv_demo.py --workspace demo_workspace_cv` (defaults to `--modalities qa fa`)

---

## Other OptiConn commands

### `sensitivity`

Analyzes which parameters have the strongest impact on quality scores.

```bash
python opticonn.py sensitivity -i <data_dir> -o <output_dir> --config <baseline_config.json>
```

Flags:

- `--parameters <p1> <p2> ...`: analyze a subset of parameters (default: all)
- `--perturbation <float>`: fractional perturbation size (default: `0.1` = 10%)
- `--verbose`: more detailed output

Outputs:

- `sensitivity_analysis_results.json`
- `sensitivity_analysis_plot.png`

### `pipeline`

Advanced wrapper around the 3-step orchestrator (`scripts/run_pipeline.py`).

```bash
python opticonn.py pipeline --step all --data-dir <data_dir> -o <output_dir> --config <extraction_config.json>
```

Flags:

- `--step {01,02,03,all,analysis}`
- `-i, --input`: alias/compat input path
- `-o, --output`: output directory
- `--config`: extraction config (defaults to `configs/braingraph_default_config.json`)
- `--data-dir`: same as in Step 01
- `--cross-validated-config`: optional; converted into an extraction config for Step 01
- `--quiet`: reduce output
