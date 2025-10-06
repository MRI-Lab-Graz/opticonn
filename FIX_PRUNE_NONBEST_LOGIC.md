# Fix: --prune-nonbest Logic Error

## Problem Identified

The `--prune-nonbest` flag had a critical logic error in its original implementation:

### ❌ Original (Incorrect) Behavior
- `--prune-nonbest` was an argument to `opticonn sweep`
- Pruning happened **immediately after each wave** completed
- Deleted all non-best combinations **within that wave**
- **Problem**: This happens BEFORE cross-wave comparison!

### Why This Was Wrong

1. **During sweep**: We're collecting metrics, not selecting final candidates
2. **Wave-level pruning**: Each wave has a "best" combo, but we don't know the global best yet
3. **Premature deletion**: Could delete combos that are optimal across BOTH waves
4. **Loss of data**: Cross-validation comparison needs all wave results intact

### Example of the Problem

```
Wave 1: atlas_A_metric_X (QA=0.85) <- "best" in wave1, kept
        atlas_B_metric_Y (QA=0.82) <- deleted by --prune-nonbest
        
Wave 2: atlas_A_metric_X (QA=0.70) <- exists in wave2
        atlas_B_metric_Y (QA=0.88) <- exists in wave2

Cross-wave: atlas_B_metric_Y would be BEST overall (avg QA=0.85)
            BUT it was already deleted from wave1!
```

## ✅ Fixed Behavior

### New Implementation
- `--prune-nonbest` is now an argument to `opticonn review`
- Pruning happens **after final selection** in review step
- Only deletes non-selected combos **after** cross-wave comparison
- **Correct**: Preserves all data until we know the true global optimum

### Updated Workflow

```bash
# Step 1: Sweep - collect ALL metrics (no pruning)
opticonn sweep \
  --config configs/sweep_production_full.json \
  --data-dir data \
  --output-dir production \
  --subjects 5 \
  --verbose

# Step 2: Review - select best AND optionally prune
opticonn review \
  --output-dir production/sweep-<uuid>/optimize \
  --auto-select-best \
  --prune-nonbest  # ← NOW it's safe to prune!

# Step 3: Apply
opticonn apply \
  --data-dir data/full \
  --optimal-config production/sweep-<uuid>/optimize/selected_candidate.json
```

## Implementation Details

### Files Modified

1. **`scripts/opticonn_hub.py`**
   - Removed `--prune-nonbest` from sweep subparser
   - Added `--prune-nonbest` to review subparser
   - Implemented pruning logic after selection (lines ~214-236)
   - Prunes based on final selected atlas+metric combination

2. **`scripts/cross_validation_bootstrap_optimizer.py`**
   - Removed premature pruning logic from `run_wave_pipeline()`
   - Removed `prune_nonbest` parameter from function signature
   - Removed `--prune-nonbest` argument from argparser

3. **`CLI_REFERENCE.md`**
   - Moved `--prune-nonbest` documentation to review section
   - Updated examples to show correct usage
   - Added "Disk Space Management" section to review

### Pruning Logic (opticonn review)

```python
if args.prune_nonbest:
    best_combo_key = f"{best_dict['atlas']}_{best_dict['connectivity_metric']}"
    
    for wave_dir in wave_dirs:
        for combo_dir in combos_dir.iterdir():
            combo_key = extract_key_from_dirname(combo_dir.name)
            if combo_key != best_combo_key:
                shutil.rmtree(combo_dir)  # Delete non-optimal combo
```

## Benefits of Fix

1. **Correctness**: No premature deletion of potentially optimal candidates
2. **Data integrity**: Cross-wave validation uses complete data
3. **Logical flow**: Pruning happens AFTER selection, not before
4. **User control**: Explicit opt-in for disk cleanup at the right time
5. **Disk savings**: Still achieves goal of reducing storage (just at correct point)

## Migration Guide

### Old Command (Incorrect)
```bash
# DON'T do this anymore
opticonn sweep --data-dir data --output-dir out --prune-nonbest
```

### New Command (Correct)
```bash
# Step 1: Sweep without pruning
opticonn sweep --data-dir data --output-dir out

# Step 2: Review and prune AFTER selection
opticonn review --output-dir out/sweep-xxx/optimize --auto-select-best --prune-nonbest
```

## When to Use --prune-nonbest

**Use it when:**
- Running large sweeps (50+ combinations) with limited disk space
- You've completed selection and don't need intermediate results
- You're confident in your QA scoring methodology

**Don't use it when:**
- You might want to manually review all combinations later
- Disk space is not a concern
- You're still experimenting with QA scoring weights

## Testing Recommendations

1. **Test sweep without pruning**: Verify all combos are generated
2. **Test review with pruning**: Verify only selected combo remains
3. **Cross-wave consistency**: Ensure best combo appears in both waves before pruning
4. **Disk space measurement**: Compare before/after sizes

## Related Issues

- Fixes potential data loss in cross-validation
- Resolves logical inconsistency in workflow design
- Aligns with principle: "Sweep = collect data, Review = make decisions"

---

**Date Fixed**: October 6, 2025
**Impact**: Critical - affects correctness of optimization results
**Backward Compatibility**: Breaking change - old `--prune-nonbest` flag removed from sweep
