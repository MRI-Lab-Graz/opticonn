# 📊 Project Clarity Overview

## What We Just Did

We've created a **comprehensive documentation suite** that maps out the entire OptiConn pipeline, making it clear which scripts do what and how they connect.

### 📚 New Documentation Files

1. **`SWEEP_WORKFLOW.md`** – Visual flowchart of complete sweep execution
2. **`SCRIPT_REFERENCE.md`** – Decision tree showing which scripts to use
3. **`MINIMAL_DEMO.md`** – Step-by-step demo guide with 3 options
4. **`SCRIPTS_SUMMARY.md`** – One-page quick reference card
5. **`CLEANUP_GUIDE.md`** – Decision framework for removing unused scripts
6. **`DOCUMENTATION_INDEX.md`** – This index, maps docs to your questions

---

## The OptiConn Pipeline at a Glance

### 🟢 Core Pipeline (Always Used - 6 Scripts)
```
opticonn_hub.py
  → run_pipeline.py
    ├─ extract_connectivity_matrices.py
    ├─ aggregate_network_measures.py
    ├─ qa_cross_validator.py (prevention layers 1&2)
    └─ bootstrap_qa_validator.py (prevention layer 3)
```

**These 6 scripts handle:**
- CLI entry point
- Single parameter combo execution
- Connectivity extraction (DSI Studio)
- Network metrics computation
- QA validation + anomaly detection
- Prevention of future bugs

### 🟡 Optional: Parameter Sweep (Add 2 Scripts)
```
cross_validation_bootstrap_optimizer.py (sweep orchestrator)
  └─ sweep_utils.py (grid generation, sampling)
```

**Used only if you run `opticonn sweep`**

### 🔵 Optional: Research/Analysis (7 Scripts)
```
pareto_view.py, sensitivity_analyzer.py, statistical_analysis.py, etc.
```

**Safe to remove** (never called by main workflow)

---

## What's Actually Involved in a Sweep

### Before (Confusing):
❌ "Wait, which scripts run when I do `opticonn sweep`?"  
❌ "Why are there so many scripts doing similar things?"  
❌ "Can I delete pareto_view.py safely?"

### After (Clear):
✅ **Complete flow diagram** showing exactly which scripts run in order  
✅ **Decision tree** telling you which scripts to keep/remove  
✅ **Runnable demo** you can execute to see it yourself  
✅ **Prevention system** integrated and documented  

---

## The Minimal OptiConn (Production Setup)

If you only extract single subjects (no parameter optimization):

```
Keep:
├── opticonn_hub.py
├── run_pipeline.py
├── extract_connectivity_matrices.py
├── aggregate_network_measures.py
├── qa_cross_validator.py
└── bootstrap_qa_validator.py

Safe to remove:
├── cross_validation_bootstrap_optimizer.py (sweep-specific)
├── sweep_utils.py (sweep-specific)
├── pareto_view.py (post-analysis only)
└── 7+ research scripts (optional)
```

---

## How to Use This Documentation

### For Someone New to the Code:
1. Read: **`SCRIPTS_SUMMARY.md`** (5 min)
2. Run: **`MINIMAL_DEMO.md`** option 1 (1 min)
3. Study: **`SWEEP_WORKFLOW.md`** (10 min)

**Total: 15 minutes to understand the whole pipeline**

### For Someone Modifying the Code:
1. Open: **`SCRIPT_REFERENCE.md`** (find your target script)
2. Check: **`SWEEP_WORKFLOW.md`** (understand how it's called)
3. Use: grep to find all usages

### For Someone Cleaning Up:
1. Read: **`CLEANUP_GUIDE.md`** (decision framework)
2. Follow: 4-tier recommendations
3. Test: `opticonn.py --help` after cleanup

---

## Key Takeaways

### ✅ Prevention System is Complete
- Layer 1: Normalization bounds check (during extraction)
- Layer 2: Success detection validation (after metrics)
- Layer 3: Anomaly detection (post-sweep)

**Prevents**: Bugs like iteration 15's QA=1.0000

### ✅ Core Pipeline is Small
- **6 essential scripts** (can run everything)
- **+2 if doing sweeps** (parameter optimization)
- **+7 if doing research** (optional analysis)

### ✅ Code is Now Self-Documenting
- Clear flow diagrams in `SWEEP_WORKFLOW.md`
- Decision tree in `SCRIPT_REFERENCE.md`
- Runnable demo in `MINIMAL_DEMO.md`

### ✅ Safe to Simplify
- Identified 7+ scripts that can be removed
- No functionality loss if removed
- Cleanup guide provided

---

## What We Removed Earlier

✅ **Dash/HTML Infrastructure** (all removed):
- Removed: `/scripts/dash_app/` directory
- Removed: `dash>=2.0.0` from pyproject.toml
- Removed: `plotly>=5.0.0` from pyproject.toml
- Removed: Dash installation from install.sh

**Result**: Simplified web dependencies, core pipeline unaffected

---

## Next Steps (Your Choice)

### Option A: Clean Up the Code
```bash
# Use CLEANUP_GUIDE.md to remove unused scripts
# Keep core pipeline, optionally remove research tools
```

### Option B: Run a Demo
```bash
# Follow MINIMAL_DEMO.md to run a tiny sweep
# See which scripts are actually called
```

### Option C: Study the Flow
```bash
# Read SWEEP_WORKFLOW.md for complete diagram
# Understand how scripts connect
```

### Option D: Do Nothing
```bash
# Documentation is in place for future reference
# Core pipeline is clean and working
# Prevention system automatically active
```

---

## Files Summary

| File | Purpose | Read Time |
|------|---------|-----------|
| `SWEEP_WORKFLOW.md` | Complete flow diagram | 15 min |
| `SCRIPT_REFERENCE.md` | Decision tree + dependencies | 10 min |
| `MINIMAL_DEMO.md` | Runnable demo + explanations | 10 min |
| `SCRIPTS_SUMMARY.md` | One-page quick reference | 5 min |
| `CLEANUP_GUIDE.md` | Removal decision framework | 10 min |
| `DOCUMENTATION_INDEX.md` | This index | 2 min |

**To get complete picture**: Start with SCRIPTS_SUMMARY (5 min), then choose based on your need

---

## Recommendation

I'd suggest:

1. **Keep everything for now** (core pipeline is solid, prevention system active)
2. **Review CLEANUP_GUIDE.md** (know what *could* be removed)
3. **Run MINIMAL_DEMO.md when curious** (actually see it in action)
4. **Remove research scripts later** (when you're confident they're not needed)

The documentation is now in place—you have clarity without disruption. ✅

