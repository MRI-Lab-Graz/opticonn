# Deep Analysis: `opticonn apply` Command

## Executive Summary

The `opticonn apply` command has several **critical issues**:

1. ❌ **Unused arguments**: `--outlier-detection` and `--interactive` are defined but **never used**
2. ❌ **Inconsistent behavior**: `--quiet` is only in apply/pipeline, not in sweep/review
3. ❌ **Missing help text**: Most arguments have no descriptions
4. ❌ **Confusing semantics**: `--skip-extraction` is misleading (should be `--analysis-only`)
5. ❌ **Wrong placement**: `--outlier-detection` should be in review/QA scoring, not apply
6. ❌ **Unclear purpose**: `--candidate-index` is poorly documented

## Current Implementation Analysis

### Argument Definitions

```python
# Current implementation (opticonn_hub.py lines 84-94)
p_apply = subparsers.add_parser('apply', help='Apply optimal parameters to full dataset')
p_apply.add_argument('-i', '--data-dir', required=True)                    # ✅ Used
p_apply.add_argument('--optimal-config', required=True)                    # ✅ Used
p_apply.add_argument('-o', '--output-dir', default='analysis_results')     # ✅ Used
p_apply.add_argument('--outlier-detection', action='store_true')          # ❌ NEVER USED
p_apply.add_argument('--skip-extraction', action='store_true')            # ✅ Used (but confusing)
p_apply.add_argument('--interactive', action='store_true')                # ❌ NEVER USED
p_apply.add_argument('--candidate-index', type=int, default=1)            # ✅ Used
p_apply.add_argument('--quiet', action='store_true')                      # ✅ Used
p_apply.add_argument('--no-emoji', action='store_true', help='...')       # ✅ Used
```

### What Actually Happens

1. **Loads optimal config** (JSON file from review step)
2. **Checks if it's a list or dict**:
   - **List** (optimal_combinations.json): Ranks by score, picks candidate by index
   - **Dict** (cross-validated config): Uses directly
3. **Creates extraction config** from selected candidate
4. **Calls `run_pipeline.py`** with:
   - `--step analysis` if `--skip-extraction` is True
   - `--step all` otherwise
5. **Passes `--quiet` and `--no-emoji` flags** to run_pipeline.py

### Dead Code Discovery

```python
# These arguments are DEFINED but NEVER REFERENCED in apply implementation:
args.outlier_detection  # ❌ Never used anywhere
args.interactive        # ❌ Never used anywhere
```

The implementation **completely ignores** these flags!

## Semantic Issues

### 1. `--skip-extraction` is Misleading

**Current behavior:**
- If True: Runs only step "analysis" (Step 03)
- If False: Runs "all" steps (01 connectivity extraction + 02 optimization + 03 analysis)

**Problem:**
- Name suggests "skip" = don't extract
- But what it really means: "I already extracted, just analyze"
- Users would think it's like `--dry-run` (preview without action)

**Better name:** `--analysis-only` or `--use-existing-extraction`

### 2. `--outlier-detection` Doesn't Belong Here

**Current location:** `opticonn apply` (final dataset analysis)

**Problem:**
- Outlier detection should happen during **parameter selection** (review step)
- Affects which candidate is "best" based on QA scoring
- Not something you toggle during final application

**Correct location:** `opticonn review` (or built into QA scoring methodology)

**Evidence from code:**
```python
# optimal_selection.py already has outlier detection in QA scoring:
# Line 347: "Extreme outlier detection (statistical quality, not analysis bias)"
# Line 702: "Outlier Detection: Removal of extreme scores that may indicate..."
```

It's **already in the QA methodology**, so this flag is redundant!

### 3. `--interactive` Purpose Unknown

**Current state:**
- Defined as argument
- Never used in code
- No documentation of what it should do

**Possible intentions:**
- Launch interactive plots after analysis?
- Prompt user for confirmation?
- Open Dash dashboard for final results?

**Reality:** Dead code that should be removed or implemented

### 4. `--candidate-index` is Unclear

**Current behavior:**
- If optimal-config is a list, ranks by score and picks index N
- Default is 1 (best candidate)

**Problem:**
- No help text explaining this
- Users won't understand when/why to change it
- Shouldn't be needed if review step did its job

**Better approach:**
- Review step should output `selected_candidate.json` (single candidate)
- Apply should just use it (no index needed)
- If user wants different candidate, re-run review

## Inconsistency: `--quiet` Flag

**Has --quiet:**
- ✅ `opticonn apply`
- ✅ `opticonn pipeline`

**Missing --quiet:**
- ❌ `opticonn sweep` (has `--verbose` instead)
- ❌ `opticonn review`

**Issue:** Inconsistent verbosity control across commands

**Recommendation:**
- Use `--verbose` for detailed output (sweep, review, apply)
- Use `--quiet` for minimal output (all commands)
- Or standardize on one approach

## What `run_pipeline.py` Actually Supports

From grep analysis, `run_pipeline.py` accepts:
- ✅ `--step` (01, 02, 03, all, analysis)
- ✅ `--data-dir` / `--input`
- ✅ `--output`
- ✅ `--extraction-config`
- ✅ `--cross-validated-config`
- ✅ `--quiet`
- ✅ `--no-emoji`
- ✅ `--dry-run`

**Not supported:**
- ❌ `--outlier-detection`
- ❌ `--interactive`

So even if `opticonn apply` passed these flags, `run_pipeline.py` would ignore them!

## Proper Workflow Design

### Current (Problematic) Workflow
```bash
# Step 1: Sweep
opticonn sweep -i data -o sweep_out

# Step 2: Review (produces list of candidates)
opticonn review -o sweep_out/optimize --auto-select-best

# Step 3: Apply (has to pick from list using --candidate-index)
opticonn apply \
  -i data/full \
  --optimal-config sweep_out/optimize/selected_candidate.json \
  --candidate-index 1 \
  --outlier-detection \  # Does nothing!
  --interactive          # Does nothing!
```

### Correct (Streamlined) Workflow
```bash
# Step 1: Sweep - Generate candidates
opticonn sweep -i data -o sweep_out

# Step 2: Review - Select ONE candidate (with optional pruning)
opticonn review -o sweep_out/optimize --auto-select-best --prune-nonbest

# Step 3: Apply - Just apply it (no selection needed)
opticonn apply \
  -i data/full \
  --optimal-config sweep_out/optimize/selected_candidate.json
```

**Key changes:**
- Review outputs **single candidate**, not list
- Apply has **no --candidate-index** (not needed)
- Apply has **no --outlier-detection** (handled in review)
- Apply has **no --interactive** (remove dead code)

## Recommended Changes

### 1. Remove Dead Arguments

```python
# REMOVE these (never used):
# p_apply.add_argument('--outlier-detection', action='store_true')
# p_apply.add_argument('--interactive', action='store_true')
```

### 2. Rename Confusing Arguments

```python
# BEFORE:
p_apply.add_argument('--skip-extraction', action='store_true')

# AFTER:
p_apply.add_argument('--analysis-only', action='store_true',
    help='Run only analysis on existing extraction outputs (skip connectivity extraction)')
```

### 3. Add Help Text to All Arguments

```python
p_apply.add_argument('-i', '--data-dir', required=True,
    help='Directory containing full dataset (.fz or .fib.gz files)')
p_apply.add_argument('--optimal-config', required=True,
    help='Path to selected_candidate.json from review step')
p_apply.add_argument('-o', '--output-dir', default='analysis_results',
    help='Output directory for final analysis results (default: analysis_results)')
p_apply.add_argument('--candidate-index', type=int, default=1,
    help='[Advanced] If optimal-config contains multiple candidates, select by index (1-based, default: 1)')
p_apply.add_argument('--quiet', action='store_true',
    help='Reduce console output (minimal logging)')
p_apply.add_argument('--no-emoji', action='store_true',
    help='Disable emoji in console output (Windows-safe)')
```

### 4. Consider Removing `--candidate-index`

If review step always outputs a **single selected candidate**, this argument becomes unnecessary:

```python
# If selected_candidate.json is always a single dict (not list):
# Then remove --candidate-index entirely
```

### 5. Add Consistency: `--verbose` Flag

```python
# Add to apply for consistency with sweep:
p_apply.add_argument('--verbose', action='store_true',
    help='Show detailed progress and DSI Studio commands')
```

### 6. Improve Help Text for Command

```python
p_apply = subparsers.add_parser('apply', 
    help='Apply optimal parameters to full dataset',
    description='''
    Apply the optimal tractography parameters (selected via review) to your complete dataset.
    This runs the full connectivity extraction and analysis pipeline with the chosen settings.
    ''')
```

## Migration Path

### Phase 1: Quick Fixes (No Breaking Changes)
1. ✅ Add help text to all arguments
2. ✅ Add `--verbose` flag
3. ✅ Improve command description

### Phase 2: Remove Dead Code (Minor Breaking)
1. ⚠️ Remove `--outlier-detection` (never worked anyway)
2. ⚠️ Remove `--interactive` (never worked anyway)
3. ⚠️ Add deprecation warning if used

### Phase 3: Semantic Improvements (Breaking)
1. 🔄 Rename `--skip-extraction` → `--analysis-only`
2. 🔄 Simplify: review always outputs single candidate
3. 🔄 Remove `--candidate-index` (no longer needed)

## Testing Recommendations

1. **Test dead arguments**: Verify `--outlier-detection` and `--interactive` do nothing
2. **Test --skip-extraction**: Confirm it runs only analysis step
3. **Test --candidate-index**: Verify it picks correct candidate from list
4. **Test --quiet**: Verify output reduction works
5. **Test error cases**: What if optimal-config is malformed?

## Summary Table

| Argument | Status | Action Required |
|----------|--------|-----------------|
| `-i, --data-dir` | ✅ Works | Add help text |
| `--optimal-config` | ✅ Works | Add help text |
| `-o, --output-dir` | ✅ Works | Add help text |
| `--outlier-detection` | ❌ Dead code | **REMOVE** |
| `--skip-extraction` | ⚠️ Confusing | **RENAME** to `--analysis-only` |
| `--interactive` | ❌ Dead code | **REMOVE** |
| `--candidate-index` | ⚠️ Unclear | Add help OR remove if unnecessary |
| `--quiet` | ✅ Works | Keep (add to sweep/review for consistency) |
| `--no-emoji` | ✅ Works | Keep |
| `--verbose` | ❌ Missing | **ADD** for consistency |

---

**Date**: October 6, 2025  
**Impact**: Critical usability issues, dead code, semantic confusion  
**Priority**: High - affects user experience and code maintainability
