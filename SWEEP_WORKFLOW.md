# OptiConn Sweep Workflow - Script Overview

## 📊 Quick Reference: Scripts Involved in Each Step

```
┌──────────────────────────────────────────────────────────────────────┐
│ USER RUNS:  opticonn sweep -i /data --output-dir studies/run1 --quick │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────┐
    │  opticonn_hub.py        │  (Entry point - parses CLI)
    │  - Validates args       │
    │  - Checks setup         │
    │  - Calls next stage     │
    └────────────┬────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────┐
    │  cross_validation_bootstrap_          │
    │  optimizer.py (Main Sweep Engine)    │
    │  ════════════════════════════════════ │
    │  - Loads extraction config            │
    │  - Builds parameter grid from sweep_  │
    │    parameters (sweep_utils.py)        │
    │  - Creates waves (bootstrap samples)  │
    │  - For each wave & combo:             │
    │    * Runs run_pipeline.py             │
    │    * Scores results                   │
    │  - Writes combo_diagnostics.csv       │
    │  - Generates reports                  │
    └────────────┬─────────────────────────┘
                 │
    ┌────────────┴──────────────────────────────────┐
    │                                               │
    ▼                                               ▼
┌─────────────────────────┐        ┌──────────────────────────────┐
│  run_pipeline.py        │        │  sweep_utils.py              │
│  ══════════════════════ │        │  ══════════════════════════ │
│  - Data extraction      │        │  - grid_product()            │
│  - DSI Studio calls     │        │  - random_sampling()         │
│  - Connectivity matrix  │        │  - lhs_sampling()            │
│  - QA scoring           │        │  - MATLAB range parsing      │
│  - Network analysis     │        │  - Parameter expansion       │
│                         │        │                              │
│  Calls:                 │        │  Used by:                    │
│  - run_parameter_sweep. │        │  - cross_validation_         │
│    py (if param sweep   │        │    bootstrap_optimizer       │
│    mode)                │        │  - run_parameter_sweep       │
│  - extract_connectivity │        │                              │
│    _matrices.py         │        │                              │
│  - aggregate_network_   │        │                              │
│    measures.py          │        │                              │
│  - qa_cross_validator.  │        │                              │
│    py                   │        │                              │
│  - bootstrap_qa_        │        │                              │
│    validator.py         │        │                              │
└─────────────────────────┘        └──────────────────────────────┘
    │                                    │
    └────────────┬─────────────────────┬─┘
                 │                     │
                 ▼                     ▼
    ┌──────────────────────────┐  ┌────────────────────────────┐
    │  extract_connectivity_   │  │  See sweep configs in:     │
    │  matrices.py             │  │  - configs/sweep_micro.json│
    │  ═════════════════════   │  │  - configs/sweep_nano.json │
    │  - Calls DSI Studio      │  │  - configs/sweep_*.json    │
    │  - Generates matrices    │  └────────────────────────────┘
    │  - Calls utils.py        │
    └──────────────────────────┘
                 │
    ┌────────────┴────────────┬──────────────────┐
    │                         │                  │
    ▼                         ▼                  ▼
┌──────────────────┐  ┌──────────────────────┐  ┌────────────────────┐
│ aggregate_       │  │ qa_cross_validator   │  │ bootstrap_qa_      │
│ network_        │  │ ═══════════════════  │  │ validator.py       │
│ measures.py     │  │ - Computes QA scores │  │ ════════════════   │
│ ═══════════════ │  │ - Consensus check    │  │ - Final QA audit   │
│ - Computes      │  │                      │  │ - Flags anomalies  │
│   network       │  │ [NEW PREVENTION      │  │ - Outputs to CSV   │
│   properties    │  │  SYSTEM: Layers 1&2] │  │                    │
│                 │  │                      │  │ [NEW PREVENTION    │
│ [SUPPORTS NEW   │  │                      │  │  SYSTEM: Layer 3]  │
│  PREVENTION     │  │                      │  │                    │
│  SYSTEM]        │  │                      │  │                    │
└──────────────────┘  └──────────────────────┘  └────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ Sweep Output Structure: │
                    │ sweep-<UUID>/           │
                    │ ├── optimize/           │
                    │ │   ├── bootstrap_qa_   │
                    │ │   │   wave_1/         │
                    │ │   │   ├── combos/     │
                    │ │   │   │   ├── sweep_  │
                    │ │   │   │   │   0001/   │
                    │ │   │   │   │   └── ... │
                    │ │   │   │   └── combo_  │
                    │ │   │   │       diagnos │
                    │ │   │   │       tics.cs │
                    │ │   │   │       v       │
                    │ │   │   └── wave_2/     │
                    │ │   └── combo_diagnostics_summary.csv
                    │ └── logs/               │
                    └─────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────┐
        │ USER RUNS: opticonn review -i ...optimize │
        │ [See next section: REVIEW WORKFLOW]       │
        └───────────────────────────────────────────┘
```

---

## 🔍 Step-by-Step Breakdown

### **Phase 1: Initialization** (`opticonn_hub.py`)
- Parse command-line arguments
- Call `validate_setup.py` to check:
  - Python dependencies installed
  - DSI Studio accessible
  - Config files valid
- Generate unique sweep UUID
- Route to `cross_validation_bootstrap_optimizer.py`

### **Phase 2: Sweep Engine** (`cross_validation_bootstrap_optimizer.py`)
- Load base config (e.g., `sweep_micro.json` for quick demo)
- Call `sweep_utils.py` functions to:
  - Parse sweep parameter ranges (e.g., `"0.05:0.05:0.20"`)
  - Generate parameter combinations via grid/random/LHS sampling
  - Create N validation waves (bootstrap subsets of data)
- For each (wave, parameter combination) pair:
  - Write derived config file
  - Run `run_pipeline.py` with that config
  - Collect results (connectivity matrices, QA scores)
  - Store in `combos/sweep_XXXX/` directory

### **Phase 3: Scoring & Validation** (Multiple scripts)
Per-combination results are scored/validated by:

1. **`extract_connectivity_matrices.py`** → Creates connectivity matrices from DTI
2. **`aggregate_network_measures.py`** → Computes graph metrics (efficiency, small-worldness, etc.)
3. **`qa_cross_validator.py`** → QA verification
   - **[NEW PREVENTION LAYER 1 & 2]** Checks for success/normalization bugs
4. **`bootstrap_qa_validator.py`** → Final anomaly detection
   - **[NEW PREVENTION LAYER 3]** Flags suspicious QA patterns

### **Phase 4: Summary & Review**
- Aggregate results into `combo_diagnostics.csv` per wave
- Optional: Generate quick quality reports
- Ready for `opticonn review` (manual or auto-selection)

---

## 📋 Config Files Involved

| Config | Purpose | When Used |
|--------|---------|-----------|
| `braingraph_default_config.json` | Full prod config (base) | Always (unless overridden) |
| `sweep_micro.json` | Ultra-minimal demo (2-4 combos) | `--quick` flag |
| `sweep_nano.json` | Tiny validation (4-6 combos) | Fast local testing |
| `sweep_probe.json` | Small probe sweep | Medium testing |
| `sweep_production_full.json` | Full production sweep | Final optimization |
| `bootstrap_optimization_config.json` | Master optimizer settings | Advanced use |
| Custom extraction config | Override params (rare) | `--extraction-config` |

---

## 🎯 Recommended Demo Commands

### **Option 1: Ultra-Fast (1-2 min, 2 combos)**
```bash
python opticonn.py sweep \
  -i /data/test_data \
  -o studies/demo \
  --quick \
  --subjects 1 \
  --no-emoji
```
Uses: `configs/sweep_micro.json`

### **Option 2: Quick Smoke Test (3-5 min, 4 combos)**
```bash
python opticonn.py sweep \
  -i /data/test_data \
  -o studies/demo \
  --extraction-config configs/sweep_nano.json \
  --subjects 2 \
  --no-emoji
```
Uses: `configs/sweep_nano.json`

### **Option 3: Trace Individual Combination**
```bash
# Run ONE parameter combination manually
python opticonn.py pipeline \
  --step all \
  --data-dir /data/test_data \
  --config configs/braingraph_default_config.json \
  --output /tmp/single_combo \
  --no-emoji --verbose
```
Uses: `run_pipeline.py` directly

---

## 🧹 Scripts You Can Probably Remove (if not doing sweeps)

- **`pareto_view.py`** – Only used to analyze sweep results (multi-objective trade-offs)
- **`sweep_utils.py`** – Only used by sweep scripts (grid generation, sampling)
- **`run_parameter_sweep.py`** – Legacy parameter grid generator (replaced by cross_validation_bootstrap_optimizer.py)

**Keep these** (used for all workflows):
- `run_pipeline.py` – Core single-combo execution
- `opticonn_hub.py` – Main CLI entry point
- `extract_connectivity_matrices.py` – Connectivity computation
- `aggregate_network_measures.py` – Network metrics
- `qa_cross_validator.py` – QA validation (+ prevention layers)
- `bootstrap_qa_validator.py` – Anomaly detection (+ prevention layer)

---

## 🔗 Prevention System Integration

The **new prevention system** integrates into the sweep pipeline at:

1. **Layer 1: Early normalization validation** → `qa_cross_validator.py`
   - Checks that QA scores are within reasonable bounds during computation

2. **Layer 2: Success detection validation** → `qa_cross_validator.py`
   - Ensures successful pipeline runs are properly flagged

3. **Layer 3: Anomaly detection** → `bootstrap_qa_validator.py`
   - Final check for suspicious patterns (like iteration 15's QA = 1.0000)

All three layers are automatically active during sweeps—no additional scripts to run!

