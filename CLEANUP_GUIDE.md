# 🧹 Codebase Cleanup Decision Guide

After recent improvements (prevention system, Dash removal), here's a practical guide to keep your codebase simple.

---

## Current Status

✅ **Recently Completed:**
- Removed Dash/HTML web infrastructure (dash_app/, dash dependency)
- Implemented 3-layer prevention system to prevent iteration-15-style bugs
- All prevention layers integrated into main pipeline

❌ **What's Left:**
- Some analysis/research scripts not used in main workflow
- Legacy parameter sweep code (replaced by new version)

---

## Decision Framework

### Question 1: Do you run parameter sweeps?

**YES** → Keep `sweep_utils.py` and `cross_validation_bootstrap_optimizer.py`  
**NO** → Can remove them (they're only used by `opticonn sweep`)

### Question 2: Do you need multi-objective trade-off analysis?

**YES** → Keep `pareto_view.py` (post-sweep analysis)  
**NO** → Remove it (only generates visualization, not needed for pipeline)

### Question 3: Do you do sensitivity/statistical research?

**YES** → Keep `sensitivity_analyzer.py`, `statistical_analysis.py`, etc.  
**NO** → Remove them (research tools, not in main workflow)

---

## Recommended Cleanup

### Tier 1: Safe to Remove Immediately ✅

These scripts are **never called** by the main workflow:

```bash
rm scripts/pareto_view.py                    # Post-sweep visualization only
rm scripts/run_parameter_sweep.py            # Legacy (replaced by cross_validation_bootstrap_optimizer.py)
rm scripts/sensitivity_analyzer.py           # Research tool
rm scripts/statistical_analysis.py           # Research tool
rm scripts/statistical_metric_comparator.py  # Research tool
rm scripts/aggregate_wave_candidates.py      # Analysis tool
rm scripts/metric_optimizer.py               # Research tool
```

**Impact**: None. Nothing uses these.

### Tier 2: Remove if Not Using Sweeps ⚠️

If you only extract single combos (not parameter optimization):

```bash
rm scripts/sweep_utils.py                    # Used only by sweep orchestrator
rm scripts/cross_validation_bootstrap_optimizer.py  # Sweep orchestrator
```

**Condition**: Only if you never run `opticonn sweep`  
**Check first**: `grep -r "cross_validation_bootstrap_optimizer\|sweep_utils" scripts/`

### Tier 3: Remove if Not Debugging 🟡

Utility scripts for validation/debugging (keep for safety):

```bash
# Optional—keep if unsure
- validate_setup.py
- pre_test_validation.py  
- verify_parameter_uniqueness.py
- json_validator.py
- check_script_compliance.py
```

### Tier 4: DO NOT Remove 🔴

**Core pipeline** (literally cannot run without these):

```
opticonn_hub.py                    # Main entry point
run_pipeline.py                    # Core executor
extract_connectivity_matrices.py   # Connectivity extraction
aggregate_network_measures.py      # Network metrics
qa_cross_validator.py              # QA validation + prevention 1&2
bootstrap_qa_validator.py          # Anomaly detection + prevention 3
```

---

## Cleanup Script (Do This Safely)

```bash
#!/bin/bash
# SAFE cleanup - removes only tier 1 scripts

cd /data/local/software/braingraph-pipeline

# Backup first!
git add -A
git commit -m "Before cleanup: remove research/analysis scripts"

# Remove tier 1 (safe)
rm -f scripts/pareto_view.py
rm -f scripts/run_parameter_sweep.py
rm -f scripts/sensitivity_analyzer.py
rm -f scripts/statistical_analysis.py
rm -f scripts/statistical_metric_comparator.py
rm -f scripts/aggregate_wave_candidates.py
rm -f scripts/metric_optimizer.py

# Verify nothing broke
python opticonn.py --help

echo "✅ Cleanup complete"
```

---

## What Each Script Actually Does

### 🔴 Core (Keep Always)

| Script | What It Does |
|--------|-------------|
| `opticonn_hub.py` | Parse CLI, route to sweep/apply/pipeline |
| `run_pipeline.py` | Execute single parameter combo (extract + score) |
| `extract_connectivity_matrices.py` | Call DSI Studio, build connectivity matrices |
| `aggregate_network_measures.py` | Compute graph properties (efficiency, density, etc.) |
| `qa_cross_validator.py` | QA validation + prevention layers 1 & 2 |
| `bootstrap_qa_validator.py` | Anomaly detection + prevention layer 3 |

### 🟡 Sweep-Related (Keep if doing sweeps)

| Script | What It Does |
|--------|-------------|
| `cross_validation_bootstrap_optimizer.py` | Orchestrate parameter sweep across bootstrap waves |
| `sweep_utils.py` | Build parameter grid (grid product, random, LHS sampling) |

### 🟢 Research/Post-Processing (Safe to Remove)

| Script | What It Does |
|--------|-------------|
| `pareto_view.py` | Analyze multi-objective trade-offs from sweep results |
| `run_parameter_sweep.py` | Legacy grid generator (replaced by cross_validation_bootstrap_optimizer.py) |
| `sensitivity_analyzer.py` | Compute parameter sensitivity (research tool) |
| `statistical_analysis.py` | Statistical comparisons (research tool) |
| `statistical_metric_comparator.py` | Compare metrics between conditions (research tool) |
| `aggregate_wave_candidates.py` | Wave aggregation analysis (research tool) |
| `metric_optimizer.py` | Optimize metrics (research tool) |

### 🔵 Utility/Debug (Optional)

| Script | What It Does |
|--------|-------------|
| `validate_setup.py` | Check Python deps, DSI Studio, configs |
| `pre_test_validation.py` | Quick validation checks |
| `verify_parameter_uniqueness.py` | Debug parameter combinations |
| `json_validator.py` | Validate JSON configs |
| `check_script_compliance.py` | Check code compliance |

---

## Before & After Comparison

### Current Structure
```
scripts/
├── Core Pipeline (6 files - KEEP)
├── Sweep Tools (2 files - keep if doing sweeps)
├── Research/Analysis (7 files - can remove)
├── Utilities (4 files - optional)
└── dash_app/ (ALREADY REMOVED ✅)
```

### Recommended Minimal Production Structure
```
scripts/
├── opticonn_hub.py               (entry point)
├── run_pipeline.py               (executor)
├── extract_connectivity_matrices.py
├── aggregate_network_measures.py
├── qa_cross_validator.py         (+ prevention 1&2)
├── bootstrap_qa_validator.py     (+ prevention 3)
├── cross_validation_bootstrap_optimizer.py (if doing sweeps)
├── sweep_utils.py                (if doing sweeps)
├── validate_setup.py             (recommended)
├── utils/                        (utility functions)
├── dsi_studio_tools/             (DSI Studio interface)
├── subcommands/                  (CLI subcommands)
└── test/                         (unit tests)
```

**Reduction**: ~25% less code, zero functionality loss

---

## Rollback Plan

If you remove something and need it back:

```bash
git log --oneline
git show <commit>:scripts/script_name.py > scripts/script_name.py
git add scripts/script_name.py
```

---

## Summary

| Action | Safe? | Why |
|--------|-------|-----|
| Remove `pareto_view.py` | ✅ YES | Only post-sweep visualization |
| Remove `run_parameter_sweep.py` | ✅ YES | Legacy, replaced by newer version |
| Remove research scripts | ✅ YES | Optional analysis tools |
| Remove `sweep_utils.py` | ⚠️ MAYBE | Only if you never do sweeps |
| Remove `cross_validation_bootstrap_optimizer.py` | ⚠️ MAYBE | Only if you never do sweeps |
| Remove validation scripts | 🟡 OPTIONAL | Safe but useful for debugging |
| Remove core pipeline scripts | 🔴 NO | Pipeline won't work |

