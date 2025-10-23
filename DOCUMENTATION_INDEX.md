# 📖 Documentation Index

This file maps all new documentation created to help you understand the OptiConn codebase.

---

## Quick Navigation

### 🎯 "I want to understand the whole pipeline"
→ Start with **`SWEEP_WORKFLOW.md`** (complete flow diagram)  
→ Then **`SCRIPT_REFERENCE.md`** (which scripts do what)

### 🧪 "I want to run a demo to see it in action"
→ Go to **`MINIMAL_DEMO.md`** (step-by-step guide)  
→ Includes 3 options: ultra-fast, minimal, or single combo

### 🗑️ "I want to clean up unused scripts"
→ Read **`CLEANUP_GUIDE.md`** (decision framework)  
→ Tells you what's safe to remove

### 📋 "I just need a quick reference"
→ Use **`SCRIPTS_SUMMARY.md`** (one-page summary)

### ⭐ "I want to understand the prevention system"
→ Check **`PREVENTION_SYSTEM.md`** (3-layer validation details)  
*(If file exists in repo)*

---

## Document Descriptions

### 1. **SWEEP_WORKFLOW.md**
**Purpose**: Complete visual flowchart of parameter sweep execution  
**Contains**:
- Flow diagram (ASCII art)
- What each script does
- Config files involved (sweep_micro, sweep_nano, etc.)
- Prevention system integration
- Demo commands

**Read when**: You want to understand the complete pipeline flow

---

### 2. **SCRIPT_REFERENCE.md**
**Purpose**: Quick decision tree for script dependencies  
**Contains**:
- Which scripts to use for each workflow (sweep/apply/pipeline)
- Complete dependency tree
- What scripts can be safely removed
- 50-word summary of each script

**Read when**: You need to understand what calls what

---

### 3. **MINIMAL_DEMO.md**
**Purpose**: Step-by-step guide to run a tiny sweep and understand each phase  
**Contains**:
- 3 demo options (30-sec, 1-min, single-combo)
- What to look for in output
- Script flow chart
- Troubleshooting
- Analysis of results

**Read when**: You want hands-on understanding (actually run code)

---

### 4. **SCRIPTS_SUMMARY.md**
**Purpose**: One-page quick reference card  
**Contains**:
- Core scripts table
- Sweep-specific scripts
- What you actually use (based on goal)
- Prevention system summary
- Safe to delete list

**Read when**: You need quick lookup

---

### 5. **CLEANUP_GUIDE.md**
**Purpose**: Decision framework for removing unused scripts  
**Contains**:
- 4-tier cleanup recommendations
- Decision questions (Do you run sweeps? Do you need analysis?)
- Safe removal commands
- Rollback plan
- Before/after codebase structure

**Read when**: You want to simplify the codebase

---

## Related Documentation (Existing in Repo)

- **`README.md`** – Main project overview (installation, workflow, examples)
- **`PREVENTION_SYSTEM.md`** – Details on the 3-layer validation system (created earlier)
- **`pyproject.toml`** – Dependencies (Dash/Plotly already removed ✅)

---

## File Relationships

```
SWEEP_WORKFLOW.md
  ├─ Shows: full flow with all scripts
  └─ References: config files, output structure
  
SCRIPT_REFERENCE.md
  ├─ Shows: decision tree + dependency tree
  └─ Cross-references: SWEEP_WORKFLOW.md
  
MINIMAL_DEMO.md
  ├─ Shows: how to run and what to observe
  ├─ Cross-references: SWEEP_WORKFLOW.md
  └─ Demonstrates: actual script invocation

SCRIPTS_SUMMARY.md
  ├─ Shows: compact table of all scripts
  ├─ Summarizes: SWEEP_WORKFLOW.md + SCRIPT_REFERENCE.md
  └─ Links to: MINIMAL_DEMO.md for next steps
  
CLEANUP_GUIDE.md
  ├─ Shows: what to remove (with decision framework)
  └─ Uses info from: SCRIPTS_SUMMARY.md
```

---

## How to Use This Documentation

### Scenario 1: "The code is too complicated, where do I start?"

```
1. Read: SCRIPTS_SUMMARY.md (5 min)
   → Get overview of what scripts exist
   
2. Run: MINIMAL_DEMO.md option 1 (1 min)
   → See the pipeline in action
   
3. Review: SWEEP_WORKFLOW.md (10 min)
   → Understand how scripts connect
   
4. Check: CLEANUP_GUIDE.md (5 min)
   → Know what you can safely remove
```

**Total time**: ~20 minutes to understand everything

---

### Scenario 2: "I want to modify the pipeline"

```
1. Read: SCRIPT_REFERENCE.md
   → Find which script handles your concern
   
2. Review: SWEEP_WORKFLOW.md
   → See how it's called and what calls it
   
3. Check: Cross-references in code
   → Use grep to find all usages
```

---

### Scenario 3: "The codebase is messy, let me clean it up"

```
1. Read: CLEANUP_GUIDE.md
   → Understand safe/unsafe removals
   
2. Answer: Decision questions
   → Do I run sweeps? Do I need analysis tools?
   
3. Execute: Cleanup script (with backup!)
   → Remove tier 1 scripts
   
4. Verify: Test main commands still work
   → opticonn sweep, opticonn apply, opticonn pipeline
```

---

## Key Insights

### 🎯 Core Pipeline is Small
Only **6 essential scripts** are always needed:
- `opticonn_hub.py`, `run_pipeline.py`, `extract_connectivity_matrices.py`
- `aggregate_network_measures.py`, `qa_cross_validator.py`, `bootstrap_qa_validator.py`

### 🧪 Sweep Adds 2 Scripts
If you do parameter sweeps, add:
- `cross_validation_bootstrap_optimizer.py`, `sweep_utils.py`

### 🔬 Research Tools Are Optional
7+ scripts for analysis/research can be removed without affecting pipeline:
- `pareto_view.py`, `sensitivity_analyzer.py`, `statistical_analysis.py`, etc.

### ⭐ Prevention System is Automatic
3-layer validation runs automatically in all workflows:
- Layer 1 & 2: `qa_cross_validator.py` (during extraction)
- Layer 3: `bootstrap_qa_validator.py` (post-sweep)

No extra scripts needed!

---

## Future Maintenance

As you add features, use this structure:

1. **Document the flow** in SWEEP_WORKFLOW.md style
2. **Create a decision tree** in SCRIPT_REFERENCE.md style
3. **Add a demo** to MINIMAL_DEMO.md if needed
4. **Update SCRIPTS_SUMMARY.md** with new tables
5. **Update CLEANUP_GUIDE.md** if safe removals change

---

## Questions?

- "What does X script do?" → `SCRIPTS_SUMMARY.md`
- "How are scripts connected?" → `SCRIPT_REFERENCE.md` (dependency tree)
- "Which script should I modify?" → `MINIMAL_DEMO.md` (trace execution)
- "Can I remove X script?" → `CLEANUP_GUIDE.md` (decision framework)
- "Show me the complete flow" → `SWEEP_WORKFLOW.md` (full diagram)

