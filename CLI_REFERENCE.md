# OptiConn CLI Reference Guide

## Command Overview

OptiConn provides a 3-step workflow for optimal tractography parameter discovery:

```bash
opticonn sweep   # Step 1: Parameter sweep on subset
opticonn review  # Step 2: Review & select optimal parameters  
opticonn apply   # Step 3: Apply to full dataset
```

---

## `opticonn sweep` - Parameter Sweep

Run parameter combinations across validation waves to find optimal settings.

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --data-dir` | Directory containing .fz or .fib.gz files for sweep |
| `-o, --output-dir` | Output directory for sweep results |

### Configuration Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Optional master sweep config (rare - prefer `--extraction-config`) | - |
| `--extraction-config` | Override extraction config for auto-generated waves | Auto-detected |
| `--quick` | Run tiny demonstration sweep (uses `configs/sweep_micro.json`) | False |

### Sweep Behavior

| Argument | Description | Default |
|----------|-------------|---------|
| `--subjects N` | Number of subjects to use for validation sweep | 3 |
| `--max-parallel N` | Max combinations to run in parallel per wave | 1 |

### Output & Reporting

| Argument | Description | Default |
|----------|-------------|---------|
| `--no-report` | Skip quick quality and Pareto reports after sweep | False |
| `--verbose` | Show DSI Studio commands and detailed progress for each combination | False |
| `--no-emoji` | Disable emoji in console output (Windows-safe) | False |

### Advanced Options

| Argument | Description | Status |
|----------|-------------|--------|
| `--auto-select` | Auto-select top candidates (legacy mode) | ⚠️ DEPRECATED - Use `opticonn review --auto-select-best` instead |
| `--no-validation` | Skip full setup validation before running sweep | Use with caution |

---

## `opticonn review` - Interactive Review & Selection

Review sweep results and select optimal parameters.

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-o, --output-dir` | Path to sweep optimize directory (e.g., `sweep-<uuid>/optimize`) |

### Review Modes

| Argument | Description | Default |
|----------|-------------|---------|
| _(default)_ | Launch interactive Dash web app at http://localhost:8050 | Interactive |
| `--auto-select-best` | Automatically select best candidate based on QA + cross-wave consistency | - |
| `--port PORT` | Port for Dash web server | 8050 |

### Disk Space Management

| Argument | Description | Default |
|----------|-------------|---------|
| `--prune-nonbest` | Delete non-optimal combo outputs after selection to save disk space | False |

### Output Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--no-emoji` | Disable emoji in console output (Windows-safe) | False |

**💡 Tip**: Use `--prune-nonbest` with `--auto-select-best` for large sweeps to conserve disk space!

### Example Usage

```bash
# Interactive review (opens web UI)
opticonn review --output-dir sweep-abc123/optimize

# Automated selection with disk space cleanup
opticonn review --output-dir sweep-abc123/optimize --auto-select-best --prune-nonbest
```

**Output**: Creates `selected_candidate.json` with chosen parameters

---

## `opticonn apply` - Apply to Full Dataset

Apply selected optimal parameters to complete dataset.

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --data-dir` | Directory containing full dataset (.fz or .fib.gz files) |
| `--optimal-config` | Path to `selected_candidate.json` from review step |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-o, --output-dir` | Output directory for final analysis results | `analysis_results` |
| `--analysis-only` | Run only analysis on existing extraction outputs (skip connectivity extraction) | False |
| `--candidate-index N` | [Advanced] If optimal-config has multiple candidates, select by 1-based index | 1 |
| `--verbose` | Show detailed progress and DSI Studio commands | False |
| `--quiet` | Reduce console output (minimal logging) | False |
| `--no-emoji` | Disable emoji in console output (Windows-safe) | False |

### Deprecated Arguments

| Argument | Status | Replacement |
|----------|--------|-------------|
| `--skip-extraction` | ⚠️ DEPRECATED | Use `--analysis-only` instead |
| `--outlier-detection` | ❌ REMOVED | Built into QA scoring (review step) |
| `--interactive` | ❌ REMOVED | Never implemented |

### Example Usage

```bash
# Basic apply - full extraction + analysis
opticonn apply \
  --data-dir /path/to/full/dataset \
  --optimal-config sweep-abc123/optimize/selected_candidate.json \
  --output-dir final_results

# Analysis only (if extraction already done)
opticonn apply \
  --data-dir /path/to/full/dataset \
  --optimal-config sweep-abc123/optimize/selected_candidate.json \
  --output-dir final_results \
  --analysis-only

# Verbose mode for debugging
opticonn apply \
  --data-dir /path/to/full/dataset \
  --optimal-config sweep-abc123/optimize/selected_candidate.json \
  --output-dir final_results \
  --verbose
```

---

## `opticonn pipeline` - Legacy Direct Pipeline

Run full pipeline with specific parameters (no sweep).

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --data-dir` | Input directory with .fz or .fib.gz files |
| `--step` | Pipeline step to run: `01`, `02`, `03`, `all` |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-o, --output` | Output directory | `pipeline_output` |
| `--config` | Path to extraction config JSON | Auto-detected |

**Note**: This is the original pipeline mode. For parameter optimization, use the 3-step workflow instead.

---

## Complete Workflow Example

```bash
# Step 1: Run parameter sweep on 3 subjects
opticonn sweep \
  --config configs/sweep_nano.json \
  --data-dir data/validation_subset \
  --output-dir my_sweep \
  --subjects 3 \
  --verbose

# Step 2: Review results and auto-select best (with cleanup)
opticonn review \
  --output-dir my_sweep/sweep-<uuid>/optimize \
  --auto-select-best \
  --prune-nonbest

# Step 3: Apply to full dataset (46 subjects)
opticonn apply \
  --data-dir data/full_dataset \
  --optimal-config my_sweep/sweep-<uuid>/optimize/selected_candidate.json \
  --output-dir final_analysis \
  --outlier-detection
```

---

## Quick Reference

### Fast Validation (< 5 minutes)
```bash
opticonn sweep --quick --data-dir data --output-dir test --subjects 1
```

### Production Sweep (comprehensive)
```bash
opticonn sweep \
  --config configs/sweep_production_full.json \
  --data-dir data \
  --output-dir production \
  --subjects 5 \
  --max-parallel 2 \
  --verbose

# Then review with cleanup
opticonn review \
  --output-dir production/sweep-<uuid>/optimize \
  --auto-select-best \
  --prune-nonbest
```

### Clean Output (no emoji, for logs)
```bash
opticonn sweep --quick --data-dir data --output-dir test --no-emoji
```

---

## Configuration Files

| File | Purpose | Combinations |
|------|---------|--------------|
| `sweep_nano.json` | Ultra-fast validation | 4 (< 5 min) |
| `sweep_micro.json` | Quick testing | 8 (10-30 min) |
| `sweep_production_full.json` | Comprehensive sweep | 50 (hours) |

See `configs/` directory for all available configurations.

---

## Tips & Best Practices

1. **Start Small**: Use `--quick` or `sweep_nano.json` for initial testing
2. **Save Disk Space**: Add `--prune-nonbest` to `opticonn review` for large sweeps
3. **Parallel Processing**: Use `--max-parallel 2` or `4` to speed up sweeps
4. **Verbose Debugging**: Add `--verbose` to see exact DSI Studio commands
5. **Cross-Wave Validation**: Default 2 waves ensures reproducibility
6. **Review Before Apply**: Always review results before applying to full dataset

---

## Deprecated Features

| Feature | Status | Alternative |
|---------|--------|-------------|
| `--auto-select` (in sweep) | ⚠️ DEPRECATED | Use `opticonn review --auto-select-best` |
| `opticonn analyze` | ⚠️ DEPRECATED | Use `opticonn apply` |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NO_EMOJI` | Set to `1` to disable emoji globally |
| `PYTHONUNBUFFERED` | Set to `1` for real-time output in logs |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 130 | User interrupted (Ctrl+C) |

---

For more details, see:
- `WORKFLOW.md` - Complete 3-step workflow guide
- `CHANGELOG_QA_ENHANCEMENTS.md` - Recent improvements
- `configs/qa_scoring_enhanced.py` - QA methodology
