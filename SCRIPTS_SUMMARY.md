# 📋 OptiConn Scripts Summary - Quick Reference Card

## 🎯 Core Scripts (Always Used)

| Script | Purpose | Runs When |
|--------|---------|-----------|
| `opticonn_hub.py` | Main CLI entry point | Every command |
| `run_pipeline.py` | Single param combo execution | sweep / apply / pipeline commands |
| `extract_connectivity_matrices.py` | DSI Studio wrapper | Every combo extraction |
| `aggregate_network_measures.py` | Network metrics computation | Every combo |
| `qa_cross_validator.py` | QA validation + prevention layers 1&2 | Every combo + post-sweep |
| `bootstrap_qa_validator.py` | Anomaly detection + prevention layer 3 | Post-sweep |

## 🧪 Sweep-Specific Scripts

| Script | Purpose | When Needed |
|--------|---------|-------------|
| `cross_validation_bootstrap_optimizer.py` | Sweep orchestrator | `opticonn sweep` command |
| `sweep_utils.py` | Grid/sampling utilities | Used by sweep orchestrator |
| `run_parameter_sweep.py` | Legacy grid generator | **Can remove** (replaced by above) |
| `pareto_view.py` | Multi-objective analysis | **Can remove** (post-processing only) |

## 🛠️ Utility/Setup Scripts

| Script | Purpose | Optional? |
|--------|---------|-----------|
| `validate_setup.py` | Environment pre-flight check | Optional but useful |
| `pre_test_validation.py` | Quick validation | Optional |
| `verify_parameter_uniqueness.py` | Parameter debugging | Optional |
| `json_validator.py` | Config validation | Optional |

## 📊 Analysis Scripts (Research Only)

| Script | Purpose | Optional? |
|--------|---------|-----------|
| `sensitivity_analyzer.py` | Parameter sensitivity | **Can remove** |
| `statistical_analysis.py` | Statistical comparisons | **Can remove** |
| `statistical_metric_comparator.py` | Metric comparisons | **Can remove** |
| `metric_optimizer.py` | Optimization research | **Can remove** |
| `aggregate_wave_candidates.py` | Wave aggregation analysis | **Can remove** |

---

## 🚀 What You Actually Use (Based on User Goal)

### Goal: "Find optimal parameters"
```
opticonn sweep -i /data -o results --quick

Uses: (in order)
1. opticonn_hub.py
2. validate_setup.py
3. cross_validation_bootstrap_optimizer.py
   ├─ sweep_utils.py
   ├─ run_pipeline.py (per combo)
   │  ├─ extract_connectivity_matrices.py
   │  ├─ aggregate_network_measures.py
   │  └─ qa_cross_validator.py
   └─ bootstrap_qa_validator.py

Result: combo_diagnostics.csv + selected_candidate.json
```

### Goal: "Apply known parameters to full dataset"
```
opticonn apply -i /data --optimal-config best.json -o results

Uses: (in order)
1. opticonn_hub.py
2. run_pipeline.py (all subjects)
   ├─ extract_connectivity_matrices.py
   ├─ aggregate_network_measures.py
   └─ qa_cross_validator.py
   └─ bootstrap_qa_validator.py

Result: Connectivity matrices for all subjects
```

### Goal: "Test a single extraction"
```
opticonn pipeline --step all -i /data --output result

Uses: (in order)
1. opticonn_hub.py
2. run_pipeline.py
   ├─ extract_connectivity_matrices.py
   ├─ aggregate_network_measures.py
   └─ qa_cross_validator.py
   └─ bootstrap_qa_validator.py

Result: Single connectivity matrix + metrics
```

---

## ⭐ Prevention System Integration

The **3-layer prevention system** is **automatically active** in all workflows:

```
qa_cross_validator.py:
  ├─ ⭐ LAYER 1: Normalization bounds check
  │  └─ Runs during extraction
  │  └─ Flags: Non-normalized scores, out-of-range values
  │
  └─ ⭐ LAYER 2: Success detection validation  
     └─ Runs after metrics computed
     └─ Flags: Failed runs, wrong success flags

bootstrap_qa_validator.py:
  └─ ⭐ LAYER 3: Anomaly detection
     └─ Runs after all combos in wave
     └─ Flags: Suspicious patterns (all 1.0, non-uniform, etc.)

Result: All QA issues flagged in combo_diagnostics.csv
```

**You don't need to do anything—prevention runs automatically!**

---

## 🗑️ Safe to Delete

These are **not** called by the main workflow:

```bash
# Definitely safe to remove:
- pareto_view.py (post-sweep analysis only)
- run_parameter_sweep.py (legacy, replaced)
- sensitivity_analyzer.py (research tool, optional)
- statistical_analysis.py (research tool, optional)
- statistical_metric_comparator.py (research tool, optional)  
- metric_optimizer.py (research tool, optional)
- aggregate_wave_candidates.py (analysis tool, optional)

# Optional (useful but not required):
- validate_setup.py (useful for debugging)
- pre_test_validation.py (validation tool)
- verify_parameter_uniqueness.py (debugging)
- json_validator.py (debugging)
```

## ⚠️ DO NOT Delete

These are **critical** for the pipeline:

```bash
MUST KEEP:
- opticonn_hub.py (entry point)
- run_pipeline.py (core executor)
- extract_connectivity_matrices.py (extraction)
- aggregate_network_measures.py (metrics)
- qa_cross_validator.py (validation + prevention 1&2)
- bootstrap_qa_validator.py (prevention 3)

IF DOING SWEEPS, ALSO KEEP:
- cross_validation_bootstrap_optimizer.py (sweep engine)
- sweep_utils.py (grid generation)
```

---

## 📚 Documentation Files Created

After cleanup, refer to these for understanding:

1. **`SWEEP_WORKFLOW.md`** – Detailed flow of sweep execution
2. **`SCRIPT_REFERENCE.md`** – Decision tree for which scripts to use
3. **`MINIMAL_DEMO.md`** – Step-by-step demo to understand the pipeline
4. **`PREVENTION_SYSTEM.md`** – Details on the 3-layer validation system (if exists)

---

## 🎓 The Minimal Production Setup

If you only do **single extractions** (no sweeps):

```
Keep only:
├── opticonn.py / opticonn_hub.py
├── run_pipeline.py
├── extract_connectivity_matrices.py
├── aggregate_network_measures.py
├── qa_cross_validator.py
└── bootstrap_qa_validator.py
```

Add scripts as needed for new features.

---

## 🔗 Dependencies Between Scripts

```
opticonn_hub.py
  → validate_setup.py
  → cross_validation_bootstrap_optimizer.py (if sweep)
  → run_pipeline.py
    → extract_connectivity_matrices.py
      → dsi_studio_tools/utils.py
    → aggregate_network_measures.py
    → qa_cross_validator.py
    → bootstrap_qa_validator.py

cross_validation_bootstrap_optimizer.py
  → sweep_utils.py
  → run_pipeline.py (called N times per wave)

pareto_view.py
  (standalone post-processing, can be removed)
```

