# Computation Integrity Validation System

**Date**: October 23, 2025  
**Purpose**: Prevent faulty optimization results from corrupting parameter selection

## Problem Statement

During Bayesian optimization iteration 15, a suspicious QA score of **1.0000** was recorded despite using only a single subject. Investigation revealed:

1. **Success Detection Bug**: DSI Studio command succeeded (returncode=0) but only partial connectivity matrices were generated (count succeeded, FA/QA/NQA failed)
2. **Normalization Bug**: Single-subject quality scores were artificially inflated to 1.0 when max==min
3. **No Validation**: Faulty results were used for best score calculation, corrupting optimization

## Solution: Automatic Integrity Validation

The system now **automatically validates** every optimization iteration and **flags faulty results** before they contaminate best score tracking.

## Validation Checks

### 1. Connectivity Matrix Generation Success ✓
**What it checks**: All requested connectivity metrics were successfully extracted

**Implementation**: 
- Reads `01_extraction/logs/extraction_summary.json`
- Verifies no failed atlases
- Confirms all connectivity.mat files exist and are non-empty

**Failure**: If count succeeds but FA/QA/NQA fail, iteration is marked FAULTY

```json
{
  "valid": false,
  "reason": "Connectivity matrix extraction failure",
  "details": "Failed atlases: 1/1. Only partial connectivity matrices generated (e.g., count succeeded but FA/QA/NQA failed)."
}
```

### 2. Quality Score Normalization Validity ✓
**What it checks**: Quality scores are not artificially inflated

**Implementation**:
- Detects when all raw scores are identical (single subject case)
- Flags if normalized scores are 1.0 despite no variation

**Failure**: Single-subject with uniform scores → marked FAULTY

```json
{
  "valid": false,
  "reason": "Artificial quality score (single subject)",
  "details": "All quality scores identical (0.4957) and normalized to 1.0 - insufficient variation for evaluation"
}
```

### 3. Output Files Validation ✓
**What it checks**: Expected output files exist and are accessible

**Implementation**:
- Verifies results directory structure
- Confirms connectivity matrices exist in atlas directories
- Checks for non-empty files

**Failure**: Missing output files → marked FAULTY

### 4. Network Metrics Sanity Checks ✓
**What it checks**: Network metrics are within expected ranges

**Implementation**:
- Validates no NaN or infinite values
- Density ∈ [0, 1]
- Clustering coefficient ∈ [0, 1]
- Other graph metrics within expected bounds

**Failure**: Out-of-range metrics → marked FAULTY

## Integration with Bayesian Optimizer

### Automatic Triggering
```python
# In bayesian_optimizer.py _evaluate_params() method
validation_result = self._validate_computation_integrity(df, iteration, output_dir)
if not validation_result['valid']:
    # Mark as faulty, return neutral score (0.0)
    # Does NOT update best_score
    return 0.0  # Neutral score
```

### Result Tracking
Each iteration now has a `faulty` flag:
```json
{
  "iteration": 15,
  "qa_score": 0.0,
  "faulty": true,
  "fault_reason": "Connectivity matrix extraction failure",
  "params": {...}
}
```

### Best Score Protection
```python
# Only valid iterations contribute to best_score
if mean_qa > self.best_score:  # Only happens if validation passed
    self.best_score = mean_qa
    self.best_params = params.copy()
```

## User-Facing Output

### During Optimization
```
❌ COMPUTATION FLAGGED AS FAULTY - Iteration 15
   Reason: Connectivity matrix extraction failure
   Details: Failed atlases: 1/1. Only partial connectivity matrices generated...
```

### Final Report
```
📈 ALL ITERATIONS (sorted by QA score):
Iter |  QA Score |   Status    |      Best Atlas       | Atlas QA | Key Parameters
  15 |    0.0000 | ❌ FAULTY   | Extraction failure    |   N/A    | tract=10000 fa=0.071 angle=70.0°

⚠️  FAULTY ITERATIONS DETECTED AND FLAGGED: 1/25
These iterations were detected as invalid and were NOT used for best score calculation:
  • Iteration 15: Connectivity matrix extraction failure
```

## Related Code Changes

### 1. extract_connectivity_matrices.py
**Added**: `_check_connectivity_files_created()` method
- Verifies all expected .connectivity.mat files exist
- Prevents false "success" when only partial extraction succeeded

### 2. metric_optimizer.py
**Fixed**: `compute_quality_scores()` normalization
- Changed single-subject scores from 1.0 to 0.5 (neutral)
- Avoids artificial score inflation

### 3. bayesian_optimizer.py
**Added**: `_validate_computation_integrity()` method
- Comprehensive validation before accepting results
- Automatic faulty result detection
- Prevents faulty results from affecting best score

## Prevention Checks

| Check | Before Fix | After Fix | Frequency |
|-------|-----------|-----------|-----------|
| Connectivity extraction success | ❌ Only checked return code | ✓ Verifies file creation | Every iteration |
| Quality score inflation | ❌ No check | ✓ Detects artificial 1.0 | Every iteration |
| Output file existence | ❌ No check | ✓ Verifies directory structure | Every iteration |
| Network metric validity | ❌ No check | ✓ Range validation | Every iteration |

## How to Use

### Default Behavior (Automatic)
Just run Bayesian optimization normally - validation happens automatically:
```bash
python scripts/bayesian_optimizer.py \
  --data-dir /path/to/data \
  --output results/ \
  --n-iterations 30
```

### Manual Validation (Debugging)
Inspect faulty iterations:
```bash
# Check progress file
cat results/bayesian_optimization_progress.json | grep -A5 "faulty"

# View iteration logs
cat results/iterations/iteration_0015/02_optimization/optimized_metrics.csv
```

## Future Enhancements

- [ ] Add validation caching to prevent re-checking completed iterations
- [ ] Create validation report for each iteration
- [ ] Add user-configurable validation thresholds
- [ ] Implement recovery mechanisms for partially failed iterations
- [ ] Add validation-specific logging level

## Testing

Test cases for integrity validation:
```bash
# Test with single subject (should be flagged as artificial)
pytest tests/test_integrity_checks.py::test_single_subject_detection

# Test with partial extraction failure (should be flagged)
pytest tests/test_integrity_checks.py::test_extraction_failure_detection

# Test with out-of-range metrics (should be flagged)
pytest tests/test_integrity_checks.py::test_metric_range_validation
```

## References

- **Issue**: Iteration 15 suspicious QA score of 1.0000
- **Root Causes**: 
  1. Success detection only checked return code
  2. QA normalization artificially inflated single-subject scores
- **Fix Date**: October 23, 2025
- **PR**: [Link to PR]

---

**Contact**: karl.koschutnig@uni-graz.at  
**Lab**: MRI-Lab Graz
