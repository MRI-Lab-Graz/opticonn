# 🧪 Minimal Reproducible Demo: Understanding OptiConn Sweep

This guide walks you through a tiny sweep to see which scripts are actually involved.

## Prerequisites

```bash
# 1. Make sure you have test data with at least 1 .fz file
ls -lah /path/to/test_data/*.fz

# 2. Activate environment
cd /data/local/software/braingraph-pipeline
source braingraph_pipeline/bin/activate

# 3. Validate setup is working
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

---

## Option 1: Ultra-Minimal (30-60 seconds)

**Config**: `sweep_micro.json` (2 combos, 1 wave)

```bash
# Run sweep
python opticonn.py sweep \
  -i /path/to/test_data \
  -o /tmp/demo_sweep \
  --quick \
  --subjects 1 \
  --no-emoji \
  --verbose

# Expected output structure
tree /tmp/demo_sweep/sweep-*/optimize -L 3
```

**Scripts you'll see called:**
1. ✅ `opticonn_hub.py` (entry)
2. ✅ `validate_setup.py` (pre-flight)
3. ✅ `cross_validation_bootstrap_optimizer.py` (sweep engine)
4. ✅ `sweep_utils.py` (grid generation)
5. ✅ `run_pipeline.py` × 2 combos (execution)
6. ✅ `extract_connectivity_matrices.py` (per combo)
7. ✅ `aggregate_network_measures.py` (per combo)
8. ✅ `qa_cross_validator.py` (per combo) → **Layer 1 & 2 prevention**
9. ✅ `bootstrap_qa_validator.py` (final) → **Layer 3 prevention**

---

## Option 2: Minimal Named Configs (1-2 min)

**Config**: `sweep_nano.json` (4-6 combos, 1 wave)

```bash
python opticonn.py sweep \
  -i /path/to/test_data \
  -o /tmp/demo_sweep_2 \
  --extraction-config configs/sweep_nano.json \
  --subjects 2 \
  --no-emoji \
  --verbose
```

---

## Option 3: Trace a Single Combo (no sweep overhead)

If you just want to understand the core pipeline without the sweep complexity:

```bash
# Single subject, single combo
python opticonn.py pipeline \
  --step all \
  --data-dir /path/to/test_data \
  --config configs/braingraph_default_config.json \
  --output /tmp/single_test \
  --no-emoji \
  --verbose

# This calls:
# 1. run_pipeline.py
#    ├─ extract_connectivity_matrices.py
#    ├─ aggregate_network_measures.py
#    ├─ qa_cross_validator.py (Layer 1 & 2)
#    └─ bootstrap_qa_validator.py (Layer 3)
```

---

## What to Look For in Output

### 1. Sweep Generation Phase
```
[cross_validation_bootstrap_optimizer.py]
  Loading config: configs/sweep_micro.json
  Sweep parameters: otsu_range, fa_threshold_range, ...
  [sweep_utils.py] Building grid: 2 combinations
  [sweep_utils.py] Creating bootstrap waves: 1 wave(s)
```

### 2. Per-Combo Execution Phase
```
[run_pipeline.py] Combo 1/2: running extraction...
  [extract_connectivity_matrices.py] Calling DSI Studio...
  [extract_connectivity_matrices.py] Built connectivity matrix
  [aggregate_network_measures.py] Computing network metrics...
  [qa_cross_validator.py] ✅ QA validation (Layer 1 & 2): PASS
    - Normalization bounds: OK
    - Success flags: OK
    - No anomalies detected during validation
```

### 3. Final Validation Phase
```
[bootstrap_qa_validator.py] ✅ Anomaly detection (Layer 3): PASS
  - QA score range check: OK
  - Success rate check: OK
  - No suspicious patterns found
```

### 4. Results Summary
```
✅ Parameter sweep completed successfully!
📊 Results saved to: /tmp/demo_sweep/sweep-UUID/optimize
   └── bootstrap_qa_wave_1/
       ├── combo_diagnostics.csv (all results)
       └── combos/sweep_0001/, sweep_0002/
```

---

## Analyzing Results After Sweep

```bash
cd /tmp/demo_sweep/sweep-*/optimize

# View all tested combinations and scores
cat bootstrap_qa_wave_1/combo_diagnostics.csv | head -5

# Check for anomaly flags
grep -i "suspicious\|anomal\|WARN" bootstrap_qa_wave_1/combo_diagnostics.csv

# Count successful combos
wc -l bootstrap_qa_wave_1/combo_diagnostics.csv

# Next: auto-select best combo
python opticonn.py review -i . --no-emoji
# Outputs: selected_candidate.json
```

---

## Script Flow Chart (What calls what)

```
opticonn.py sweep ...
  ↓
[opticonn_hub.py] Parses CLI
  ↓
[validate_setup.py] Pre-flight checks
  ├─ Check Python deps
  ├─ Check DSI Studio path
  └─ Validate config files
  ↓
[cross_validation_bootstrap_optimizer.py] Main sweep engine
  ├─ [sweep_utils.py] Build parameter grid
  │  ├─ parse_matlab_range("0.05:0.05:0.20")
  │  ├─ grid_product({otsu: [0.5], fa: [0.05, 0.2], ...})
  │  └─ Returns: [combo_1, combo_2, ...]
  │
  ├─ [sweep_utils.py] Create bootstrap waves
  │  └─ Returns: [wave_1_subjects, wave_2_subjects, ...]
  │
  └─ For each (wave, combo) pair:
     ↓
     [run_pipeline.py]
       ├─ [extract_connectivity_matrices.py]
       │  └─ DSI Studio → connectivity matrix
       │
       ├─ [aggregate_network_measures.py]
       │  └─ Compute graph metrics
       │
       ├─ [qa_cross_validator.py]
       │  ├─ ⭐ LAYER 1: Normalization bounds check
       │  └─ ⭐ LAYER 2: Success flag detection
       │
       └─ [bootstrap_qa_validator.py]
          └─ ⭐ LAYER 3: Anomaly detection (suspicious patterns)
             └─ Outputs diagnostic flags to CSV

   ↓ (after all combos)
   
   [bootstrap_qa_validator.py] Final anomaly detection
   └─ Flags any suspicious patterns across wave

   ↓
   
   combo_diagnostics.csv (per wave)
   ├─ Combo ID, Parameters, QA scores
   ├─ Success flags, Network metrics
   ├─ Anomaly flags (Layer 3)
   └─ (ready for opticonn review)

Result: /tmp/demo_sweep/sweep-UUID/optimize/
```

---

## Quick Reference: What Each Prevention Layer Does

### ⭐ Layer 1: Normalization Bounds Check (qa_cross_validator.py)
Runs **during** connectivity extraction, validates:
- QA score range is reasonable (not 0 or 1.0)
- Normalization applied correctly
- Individual QA values within bounds

**Prevents**: Normalization bugs → fake perfect scores

### ⭐ Layer 2: Success Detection Check (qa_cross_validator.py)
Runs **after** metrics computed, validates:
- Success flag properly set
- Failed extraction caught early
- Pipeline status flags accurate

**Prevents**: Treating failed runs as successful

### ⭐ Layer 3: Anomaly Detection (bootstrap_qa_validator.py)
Runs **after** all wave combos complete, flags:
- Suspicious QA patterns (too uniform, all 1.0, etc.)
- Inconsistent success rates
- Outlier parameter combinations

**Prevents**: Selecting obviously broken combos (like iteration 15)

---

## Troubleshooting Demo

### "DSI Studio not found"
```bash
# Check path
export DSI_STUDIO_PATH=/path/to/dsi_studio
python scripts/validate_setup.py --config configs/braingraph_default_config.json
```

### "No .fz files found"
```bash
# Demo data must have at least 1 diffusion image
ls /path/to/test_data/*.fz
# If empty: use your own data or create a symlink to real data
```

### "Sweep runs forever"
```bash
# Check if DSI Studio is actually running
ps aux | grep dsi_studio

# If stuck, Ctrl+C and re-run with --subjects 1 (1 subject only)
```

### "Anomaly detector flagged my results"
```bash
# This is expected! View what it flagged
cat bootstrap_qa_wave_1/combo_diagnostics.csv | grep WARN

# Review the flagged combos
cat combos/sweep_XXXX/diagnostics.json | python -m json.tool

# Note: Flags are **warnings**, not failures
#       You can still use flagged combos if you understand the warning
```

---

## Next Steps After Understanding the Flow

1. **Remove pareto_view.py** if you don't need multi-objective analysis
2. **Remove sweep_utils.py** if you only do single combos (not sweeps)
3. **Keep everything else** (core pipeline is small & essential)
4. **Trust the prevention layers** (3-layer system prevents future iteration-15-style bugs)

