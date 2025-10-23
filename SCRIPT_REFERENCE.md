# Quick Decision Tree: Which Scripts Am I Using?

## 🎯 Choose Your Workflow

### ✅ **I want to optimize parameters (find best settings)**
```
Use: opticonn sweep
Calls: cross_validation_bootstrap_optimizer.py
  ├── sweep_utils.py (grid generation)
  ├── run_pipeline.py (per combo)
  │  ├── extract_connectivity_matrices.py
  │  ├── aggregate_network_measures.py
  │  └── qa_cross_validator.py
  │      [PREVENTS: Layer 1 & 2]
  └── bootstrap_qa_validator.py
      [PREVENTS: Layer 3]

Then: opticonn review
  └── Auto-select best combo

Output: selected_candidate.json
```

### ✅ **I want to apply known-good parameters to my full dataset**
```
Use: opticonn apply
Calls: run_pipeline.py (full dataset)
  ├── extract_connectivity_matrices.py
  ├── aggregate_network_measures.py
  ├── qa_cross_validator.py
  │  [PREVENTS: Layer 1 & 2]
  └── bootstrap_qa_validator.py
      [PREVENTS: Layer 3]

Output: Analysis-ready connectivity matrices
```

### ✅ **I want to run a single connectivity extraction (manual testing)**
```
Use: opticonn pipeline --step all
Calls: run_pipeline.py
  ├── extract_connectivity_matrices.py
  ├── aggregate_network_measures.py
  ├── qa_cross_validator.py
  │  [PREVENTS: Layer 1 & 2]
  └── bootstrap_qa_validator.py
      [PREVENTS: Layer 3]

Output: Single connectivity matrix + metrics
```

---

## 📊 Script Dependency Tree

```
ENTRY POINTS (User runs these):
├── opticonn_hub.py (main CLI)
├── run_pipeline.py (single combo pipeline)
└── opticonn.py (legacy entry, routes to opticonn_hub.py)

USED BY SWEEPS ONLY:
├── cross_validation_bootstrap_optimizer.py
│   └── sweep_utils.py (grid building)
├── run_parameter_sweep.py (legacy)
└── pareto_view.py (post-sweep analysis)

USED BY ALL WORKFLOWS (extract & score):
├── extract_connectivity_matrices.py
│   └── dsi_studio_tools/utils.py
├── aggregate_network_measures.py
├── qa_cross_validator.py ⭐ [PREVENTION: Layer 1 & 2]
└── bootstrap_qa_validator.py ⭐ [PREVENTION: Layer 3]

VALIDATION & SETUP:
├── validate_setup.py
├── pre_test_validation.py
├── scripts/test/** (unit tests)
└── utils/* (utility functions)

ADVANCED ANALYSIS:
├── sensitivity_analyzer.py (not auto-run)
├── statistical_analysis.py (not auto-run)
├── statistical_metric_comparator.py (not auto-run)
└── metric_optimizer.py (not auto-run)
```

---

## 🗑️ Safe to Remove (if not doing parameter sweeps)

| Script | Used By | Safe? |
|--------|---------|-------|
| `pareto_view.py` | Manual sweep analysis | ✅ Yes (post-processing only) |
| `sweep_utils.py` | Parameter grid generation | ✅ Yes (sweep-only utilities) |
| `run_parameter_sweep.py` | Legacy sweep runner | ✅ Yes (replaced by cross_validation_bootstrap_optimizer.py) |
| `sensitivity_analyzer.py` | Research analysis | ✅ Yes (optional analysis) |
| `statistical_analysis.py` | Research analysis | ✅ Yes (optional analysis) |
| `statistical_metric_comparator.py` | Research analysis | ✅ Yes (optional analysis) |
| `metric_optimizer.py` | Research optimization | ✅ Yes (optional) |

---

## ⚠️ DO NOT Remove (core pipeline)

| Script | Purpose | Critical? |
|--------|---------|-----------|
| `opticonn_hub.py` | Main entry point | 🔴 **YES** |
| `run_pipeline.py` | Single combo/full dataset execution | 🔴 **YES** |
| `cross_validation_bootstrap_optimizer.py` | Sweep orchestration | 🔴 **YES** (if doing sweeps) |
| `extract_connectivity_matrices.py` | Connectivity extraction | 🔴 **YES** |
| `aggregate_network_measures.py` | Network metrics | 🔴 **YES** |
| `qa_cross_validator.py` | QA validation + **Layer 1 & 2 prevention** | 🔴 **YES** |
| `bootstrap_qa_validator.py` | Anomaly detection + **Layer 3 prevention** | 🔴 **YES** |
| `validate_setup.py` | Environment validation | 🟡 *Useful* |

---

## 🧪 Demo: Minimal Sweep to Understand Flow

```bash
# Navigate to repo
cd /data/local/software/braingraph-pipeline

# Activate environment
source braingraph_pipeline/bin/activate

# Run minimal sweep (2 parameter combinations, 1 subject)
python opticonn.py sweep \
  -i /path/to/test_data/with_some_fz_files \
  -o demo_results \
  --quick \
  --subjects 1 \
  --no-emoji \
  --verbose

# Monitor output
cd demo_results/sweep-*/optimize/
head -20 bootstrap_qa_wave_1/combo_diagnostics.csv
tail -20 bootstrap_qa_wave_1/combo_diagnostics.csv
```

**What you'll see:**
1. `validate_setup.py` runs first
2. `cross_validation_bootstrap_optimizer.py` launches
3. For each combo: `run_pipeline.py` called
   - DSI Studio extracts connectivity
   - Network metrics computed
   - QA validation runs (Layers 1 & 2)
4. Results written to `combos/sweep_XXXX/`
5. Final summary: `bootstrap_qa_wave_1/combo_diagnostics.csv`
6. Anomaly detection runs (Layer 3)

**You can now understand:**
- Which parameters were tested (sweep grid)
- How many combos ran
- Which QA checks were applied
- What the anomaly detector flagged

---

## 🎓 What Each Script Does (50-word summary)

### Core Pipeline
- **`opticonn_hub.py`** – CLI dispatcher; parses args and routes to appropriate workflow (sweep/apply/pipeline)
- **`run_pipeline.py`** – Executes single parameter combo; coordinates extraction, scoring, QA validation
- **`extract_connectivity_matrices.py`** – Calls DSI Studio to build connectivity matrices from DTI data
- **`aggregate_network_measures.py`** – Computes graph properties (efficiency, small-worldness, density) on matrices

### Sweep-Specific
- **`cross_validation_bootstrap_optimizer.py`** – Sweep orchestrator; generates param combos, creates bootstrap waves, runs each combo
- **`sweep_utils.py`** – Grid building utilities (Cartesian product, random/LHS sampling, MATLAB range parsing)
- **`run_parameter_sweep.py`** – Legacy parameter grid generator (can remove if not needed)

### Validation & Prevention
- **`qa_cross_validator.py`** – QA checks and prevention layers 1 & 2 (normalization bounds, success detection)
- **`bootstrap_qa_validator.py`** – Anomaly detection and prevention layer 3 (flags suspicious patterns)
- **`validate_setup.py`** – Pre-flight checks (deps, DSI Studio, config validity)

### Analysis (Optional)
- **`pareto_view.py`** – Post-sweep: analyzes multi-objective trade-offs (cost vs. quality vs. density)
- **`sensitivity_analyzer.py`** – Parameter sensitivity analysis (optional research tool)
- **`statistical_analysis.py`** – Statistical comparisons (optional research tool)

