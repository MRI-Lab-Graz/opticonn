# OptiConn User Guide

## Overview

OptiConn is a command-line interface (CLI) tool designed to automate the optimization of structural connectome generation. It wraps DSI Studio and provides a layer of intelligent parameter selection.

## Conceptual workflow (what OptiConn does)

OptiConn is designed around a simple, reproducible loop:

**Parameter space exploration → objective + QC/QA scoring → robust selection → freeze config → downstream graph analysis**

Concretely:

1. **Fix the analysis target**: choose an atlas (node definition), an edge definition (how streamlines become weights), and an objective metric that reflects your downstream goal (graph-theoretical analyses).
2. **Explore tractography parameters**: evaluate many tractography parameter sets (via Bayesian tuning or a systematic sweep) on a pilot dataset.
3. **Score each candidate**:
    - **QC/QA (Quality Control/Assurance)** checks ensure outputs are plausible and usable (e.g., basic integrity/sanity of generated connectivity outputs).
    - Your **objective metric** drives what “best” means for your atlas + edge definition.
4. **Select robust parameters**: pick a winner (or a small shortlist) that performs well and is stable across resampling / subjects (bootstrap and/or cross-validation depending on the tuning mode).
5. **Freeze the configuration**: save the chosen parameter set as the configuration you will reuse.
6. **Apply to full data**: generate analysis-ready connectomes for your full cohort using the frozen config, then proceed with graph-theoretical methods.

### Terminology note: “QA” can mean two different things

- In OptiConn CLI flags like `--modalities qa`, `qa` refers to **Quantitative Anisotropy** (a DSI Studio-derived scalar used as a tracking/contrast modality).
- In this documentation, **QC/QA** refers to **Quality Control/Assurance** checks.

## Installation

See the main `README.md` for installation instructions. Ensure you have activated the virtual environment:

```bash
source braingraph_pipeline/bin/activate
```

## Demo Data

For testing purposes, we recommend using the OpenNeuro dataset **ds003138**:
[https://openneuro.org/datasets/ds003138/versions/1.0.1](https://openneuro.org/datasets/ds003138/versions/1.0.1)

## Core Commands

The `opticonn.py` script is the main entry point. It supports several subcommands.

### 1. `bayesian` - Bayesian Optimization

The recommended method for finding optimal parameters.

```bash
python opticonn.py bayesian -i <INPUT_DIR> -o <OUTPUT_DIR> [options]
```

**Arguments:**
- `-i, --input-dir`: Path to directory containing `.fib.gz` or `.fz` files.
- `-o, --output-dir`: Directory where results will be stored.
- `--config`: (Optional) Path to a JSON configuration file. Defaults to `configs/braingraph_default_config.json`.
- `--n-iterations`: (Optional) Number of optimization iterations. Default is usually sufficient (20-50).
- `--sample-subjects`: (Recommended) Use a different random subject for each iteration to prevent overfitting.
- `--no-emoji`: (Optional) Disable emoji output (useful for logging or Windows).

**Output:**
- `bayesian_optimization_results.json`: The best parameters found.
- `iterations/`: A folder containing the results of every iteration.

### 2. `sweep` - Parameter Sweep (Grid Search)

A systematic exploration of the parameter space. Useful for baselines.

```bash
python opticonn.py sweep -i <INPUT_DIR> -o <OUTPUT_DIR> [options]
```

**Arguments:**
- `-i, --input-dir`: Path to directory containing `.fib.gz` or `.fz` files.
- `-o, --output-dir`: Directory where results will be stored.
- `--quick`: Run a very small sweep for testing purposes.
- `--subjects`: Number of subjects to use for the sweep (default: 3).

**Output:**
- A structured directory with `optimize/` containing results for each wave.

### 3. `review` - Review Results

Selects the best configuration from a completed optimization run.

```bash
python opticonn.py review -i <RESULT_FILE> [options]
```

**Arguments:**
- `-i, --input`: Path to `bayesian_optimization_results.json` (for Bayesian) or the `optimize` folder (for Sweep).
- `--interactive`: (Sweep only) Launch a Dash web app to explore the results.

### 4. `apply` - Apply Configuration

Applies a selected configuration to a full dataset.

```bash
python opticonn.py apply -i <FULL_DATA_DIR> --optimal-config <CONFIG_FILE> -o <OUTPUT_DIR>
```

**Arguments:**
- `-i, --input-dir`: Directory containing the full dataset.
- `--optimal-config`: The `selected_candidate.json` or `bayesian_optimization_results.json` file.
- `-o, --output-dir`: Final output directory.

## Configuration Files

OptiConn uses JSON configuration files to control DSI Studio and the optimization process.

**Key Sections:**
- `dsi_studio_cmd`: Path to the binary.
- `atlases`: List of atlases to use.
- `sweep_parameters`: Ranges for Bayesian optimization or lists for Grid Search.

Example `sweep_parameters` for Bayesian:
```json
"sweep_parameters": {
    "fa_threshold_range": [0.05, 0.3],
    "turning_angle_range": [30.0, 90.0]
}
```

## Troubleshooting

- **DSI Studio not found**: Ensure `dsi_studio_cmd` is correct in the config or set `DSI_STUDIO_CMD` env var.
- **Memory errors**: Reduce `--max-workers` or `thread_count`.
