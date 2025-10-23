# Prevention System for Faulty Optimization Results

**Date**: October 23, 2025  
**Status**: ✅ Implemented and Tested  
**Component**: Automatic Computation Integrity Validation

## Quick Summary

**Problem**: Iteration 15 in Bayesian optimization showed a suspicious QA score of 1.0000, caused by:
1. Only checking command return code (missed partial failures)
2. Artificially inflating single-subject quality scores to 1.0

**Solution**: Automatic validation runs **before** results are accepted. Faulty results are flagged and **never contribute** to best score calculation.

## How It Works

### Automatic Detection
```
Iteration 15 completes
    ↓
Extract results (connectivity matrices, QA scores)
    ↓
✓ RUN VALIDATION (NEW)
    - Check extraction success (all files created)
    - Check score normalization (no artificial 1.0)
    - Check output files exist
    - Check metrics are in valid ranges
    ↓
Result: FAULTY ❌
    ↓
Mark as faulty, return neutral score (0.0)
    ↓
Best score NOT updated ✓
    ↓
Continue to next iteration
```

### What Gets Checked

| Check | Validates | Example Failure |
|-------|-----------|-----------------|
| **Extraction Success** | All connectivity matrices generated | Only count succeeded, FA failed |
| **Score Inflation** | No artificial 1.0 scores | Single subject with uniform scores |
| **Output Files** | Results directory structure correct | Missing .connectivity.mat files |
| **Metric Ranges** | Network metrics within valid ranges | Density > 1.0 or contains NaN |

## Usage

### Default (Automatic)
Just run Bayesian optimization normally - validation happens automatically:

```bash
python scripts/bayesian_optimizer.py \
  --data-dir /data/local/Poly/derivatives/meta/fz/ \
  --output results/ \
  --n-iterations 30
```

### Check Results
The output automatically flags faulty iterations:

```
📈 ALL ITERATIONS (sorted by QA score):
Iter |  QA Score |   Status    |      Best Atlas       | Atlas QA | Key Parameters
  15 |    0.0000 | ❌ FAULTY   | Extraction failure    |   N/A    | tract=10000 fa=0.071 angle=70.0°
  14 |    0.6234 | ✓ Valid     | FreeSurferDKT_Cortical| 0.6234   | tract=15000 fa=0.095 angle=65.0°

⚠️  FAULTY ITERATIONS DETECTED AND FLAGGED: 1/25
These iterations were detected as invalid and were NOT used for best score calculation:
  • Iteration 15: Connectivity matrix extraction failure
```

### Progress Tracking
Check `bayesian_optimization_progress.json`:

```json
{
  "all_iterations": [
    {
      "iteration": 15,
      "qa_score": 0.0,
      "faulty": true,
      "fault_reason": "Connectivity matrix extraction failure",
      "params": {...}
    }
  ]
}
```

## Test Coverage

All validation checks are tested:

```bash
PYTHONPATH=/data/local/software/braingraph-pipeline python scripts/test_integrity_checks.py
```

Tests include:
- ✅ Single subject artificial score detection
- ✅ Extraction failure detection
- ✅ Network metric range validation
- ✅ NaN/invalid value detection
- ✅ Valid computation acceptance
- ✅ Connectivity file checking

## Related Code Changes

### 1. `scripts/extract_connectivity_matrices.py`
**New Method**: `_check_connectivity_files_created()`
- Verifies all expected .connectivity.mat files exist
- Checks file sizes are non-zero
- Prevents false "success" from partial extraction

**Used in**: Every extraction operation to validate success

### 2. `scripts/metric_optimizer.py`
**Fixed**: `compute_quality_scores()` normalization
- Changed: Single-subject scores now 0.5 (neutral) instead of 1.0
- Prevents artificial score inflation
- Uses `np.full_like(quality_score, 0.5)` instead of `np.ones_like()`

### 3. `scripts/bayesian_optimizer.py`
**New Method**: `_validate_computation_integrity(df, iteration, output_dir)`
- Comprehensive multi-check validation
- Reads extraction logs, verifies files, validates metrics
- Returns `{'valid': bool, 'reason': str, 'details': str}`

**Modified**: `_evaluate_params()` method
- Added validation before accepting results
- Marks faulty results with `faulty: True` flag
- Returns 0.0 score for faulty results
- Only valid results can update best_score

## Detailed Validation Logic

### Check 1: Connectivity Extraction Success
```python
extraction_summary = read_json("01_extraction/logs/extraction_summary.json")
if extraction_summary['summary']['failed'] > 0:
    → FAULTY: "Connectivity matrix extraction failure"
```

### Check 2: Quality Score Normalization
```python
if len(set(raw_scores)) == 1 and all(s >= 0.99 for s in normalized_scores):
    → FAULTY: "Artificial quality score (single subject)"
```

### Check 3: Output Files Validation
```python
results_dir = "01_extraction/results"
for atlas_dir in results_dir.iterdir():
    if not glob("*.connectivity.mat"):
        → FAULTY: "Missing connectivity matrices"
```

### Check 4: Metric Range Validation
```python
for metric in ['density', 'clustering_coeff', 'efficiency']:
    if np.any(values < expected_min) or np.any(values > expected_max):
        → FAULTY: "Invalid network metrics"
```

## Prevention In Action

**Before Fix**:
```
Iteration 15: QA = 1.0000 → BEST SCORE UPDATED ✗
(Despite only partial connectivity extraction!)
```

**After Fix**:
```
Iteration 15: QA = 0.0000 → FAULTY ❌
Reason: Connectivity matrix extraction failure
Result: Best score NOT updated ✓
```

## Configuration

All validation is **automatic** - no configuration needed. But you can inspect:

**File**: `bayesian_optimization_progress.json`
- Lists all iterations with faulty flag
- Shows fault reasons
- Tracks only valid best scores

## Troubleshooting

### "Why was my iteration marked as faulty?"

Check the progress file for the reason:
```bash
jq '.all_iterations[] | select(.faulty == true)' bayesian_optimization_progress.json
```

Common reasons:
- **"Connectivity matrix extraction failure"**: Some metrics (FA/QA/NQA) didn't generate
- **"Artificial quality score"**: Single subject with uniform scores
- **"Missing connectivity matrices"**: Output files not found
- **"Invalid network metrics"**: Metrics outside expected ranges

### "How do I debug a faulty iteration?"

Check the iteration's extraction logs:
```bash
cat results/iterations/iteration_0015/01_extraction/logs/extraction_summary.json
```

Look for:
- Failed atlases
- Missing connectivity values (count succeeded but FA failed)
- DSI Studio error messages

### "Can I ignore/override faulty detections?"

No - the system is designed to prevent garbage results. If an iteration is marked faulty, it means something went wrong with the computation. Instead:

1. Check logs for the root cause
2. Verify input data quality
3. Ensure sufficient system resources
4. Re-run optimization with fixed configuration

## Future Enhancements

- [ ] Validation caching to skip already-checked iterations
- [ ] Recovery mechanisms for partially failed iterations
- [ ] User-configurable validation thresholds
- [ ] Detailed per-iteration validation reports
- [ ] Automatic recovery/retry for transient failures

## Testing Instructions

Run the full test suite:
```bash
cd /data/local/software/braingraph-pipeline
PYTHONPATH=. python scripts/test_integrity_checks.py
```

Run specific test:
```bash
PYTHONPATH=. python -c "from scripts.test_integrity_checks import test_extraction_failure_detection; test_extraction_failure_detection()"
```

## Summary Table

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| **Success Detection** | Return code only | File verification | ✅ Catches partial failures |
| **Score Normalization** | Single-subject = 1.0 | Single-subject = 0.5 | ✅ Prevents artificial inflation |
| **Validation** | None | 4-check validation | ✅ Automatic faulty detection |
| **Best Score Logic** | Uses all results | Uses only valid results | ✅ Protects best parameters |
| **Logging** | No faulty marking | Status column + summary | ✅ Clear visibility |

## References

- **Issue**: Iteration 15 suspicious QA score of 1.0000
- **Root Causes**: 
  - Success detection missed partial connectivity matrix failures
  - QA normalization artificially inflated single-subject scores
- **Implementation Date**: October 23, 2025
- **Implementation Status**: ✅ Complete and tested
- **Verification**: 6/6 test cases passing

---

**Developed by**: MRI-Lab Graz  
**Contact**: karl.koschutnig@uni-graz.at  
**Lab**: MRI-Lab Graz, University of Graz

