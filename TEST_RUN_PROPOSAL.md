# 🧪 Bayesian Optimization & Sensitivity Analysis - Test Run Proposal

## Overview

This test run validates the new Bayesian optimization and sensitivity analysis features. The goal is to demonstrate that these methods can efficiently find optimal tractography parameters with significantly fewer evaluations than traditional grid search.

## Prerequisites

### 1. Environment Setup
```bash
# Install with Bayesian optimization support
./install.sh
source braingraph_pipeline/bin/activate
```

### 2. Test Data Requirements

**You need DWI data with fiber bundles (.fib files) for testing.**

**Option A: Use existing test data**
If you have existing `.fib` files, organize them like this:
```
data/
├── fib_samples/
│   ├── subject01.fib
│   ├── subject02.fib
│   └── subject03.fib
```

**Option B: Create minimal test data**
For a quick smoke test, you can use any `.fib` file you have, even just one subject.

**Option C: Use example data**
If you have access to example datasets from previous runs, use those.

## Test Run Plan

### Phase 1: Sensitivity Analysis (Fast - ~2-3 hours)

**Purpose**: Identify which parameters have the most impact on QA scores.

**Command**:
```bash
opticonn sensitivity \
  -i data/fib_samples \
  -o test_results/sensitivity \
  --config configs/sweep_nano.json \
  --parameters tract_count fa_threshold min_length turning_angle step_size track_voxel_ratio connectivity_threshold
```

**Expected Output**:
- `test_results/sensitivity/sensitivity_analysis_results.json`
- `test_results/sensitivity/sensitivity_plots.png`
- Console output showing parameter rankings

**Expected Results**:
```
Parameters ranked by impact:
1. fa_threshold: HIGH (-0.085 QA per 0.01 FA)
2. tract_count: MEDIUM (+0.012 QA per 10k tracks)
3. min_length: LOW (+0.003 QA per mm)
...
```

**Timeline**: ~2-3 hours (7 parameters × ~20 min each)

### Phase 2: Bayesian Optimization (Medium - ~4-6 hours)

**Purpose**: Find optimal parameter combination using intelligent search.

**Command**:
```bash
opticonn bayesian \
  -i data/fib_samples \
  -o test_results/bayesian \
  --config configs/sweep_nano.json \
  --n-iterations 10 \
  --parameters tract_count fa_threshold min_length turning_angle step_size track_voxel_ratio connectivity_threshold
```

**Expected Output**:
- `test_results/bayesian/bayesian_optimization_results.json`
- `test_results/bayesian/optimization_plots.png`
- `test_results/bayesian/best_parameters.json`
- Progress updates every iteration

**Expected Results**:
```
Iteration 10/10 - Best QA: 0.8765
Parameters: tract_count=125000, fa_threshold=0.12, ...
Convergence achieved in 10 iterations
```

**Timeline**: ~4-6 hours (10 iterations × ~25-30 min each)

### Phase 3: Comparison with Traditional Grid Search (Optional)

**Purpose**: Demonstrate efficiency gains.

**Command**:
```bash
opticonn sweep \
  -i data/fib_samples \
  -o test_results/grid_comparison \
  --config configs/sweep_nano.json \
  --quick
```

**Expected Output**: Full grid search results for comparison

**Timeline**: ~1-2 hours (but would be much longer with larger grids)

## Validation Steps

### 1. Check Installation
```bash
# Verify scikit-optimize is installed
python -c "import skopt; print('scikit-optimize:', skopt.__version__)"

# Test CLI commands are available
opticonn --help | grep -E "(bayesian|sensitivity)"
```

### 2. Validate Sensitivity Analysis Results
```bash
# Check results file exists
ls -la test_results/sensitivity/

# View parameter rankings
cat test_results/sensitivity/sensitivity_analysis_results.json | jq '.parameter_importance'

# Verify plots were generated
ls test_results/sensitivity/*.png
```

### 3. Validate Bayesian Optimization Results
```bash
# Check convergence
cat test_results/bayesian/bayesian_optimization_results.json | jq '.convergence'

# View best parameters
cat test_results/bayesian/best_parameters.json

# Check optimization plots
ls test_results/bayesian/*.png
```

### 4. Compare with Baseline
```bash
# Compare QA scores
echo "Bayesian best QA:"
cat test_results/bayesian/best_parameters.json | jq '.qa_score'

echo "Grid search best QA:"
cat test_results/grid_comparison/aggregated_network_measures.csv | sort -k2 -n | tail -1
```

## Expected Performance Metrics

### Sensitivity Analysis
- **Runtime**: 2-3 hours
- **Evaluations**: 7-14 (one per parameter + perturbations)
- **Output**: Parameter importance ranking with effect sizes

### Bayesian Optimization
- **Runtime**: 4-6 hours for 10 iterations
- **Evaluations**: 10-15 total (5 initial + 5-10 optimization)
- **Convergence**: Should find parameters within 5-10% of optimal
- **Efficiency**: ~10x faster than equivalent grid search

### Quality Comparison
- **Grid Search**: Exhaustive but slow (days for large spaces)
- **Bayesian**: Near-optimal results with 95% time savings

## Troubleshooting

### Common Issues

**1. No .fib files found**
```
Error: No .fib files found in data/fib_samples/
```
**Solution**: Check your data directory structure and file paths.

**2. DSI Studio not found**
```
Error: DSI Studio executable not found
```
**Solution**: Update `dsi_studio_cmd` in config file to match your installation path.

**3. Memory issues**
```
Error: Out of memory during tractography
```
**Solution**: Reduce `tract_count` in config or use fewer subjects.

**4. Slow performance**
- Use `--n-iterations 5` for faster testing
- Use fewer parameters: `--parameters tract_count fa_threshold`
- Use single subject for initial testing

### Debug Mode
```bash
# Enable verbose output
opticonn sensitivity -i data/fib_samples -o test_results/sens --config configs/sweep_nano.json --verbose

# Check individual pipeline steps
opticonn run -i data/fib_samples/subject01.fib -o test_single --config configs/sweep_nano.json
```

## Success Criteria

### Sensitivity Analysis ✅
- [ ] Parameter importance file generated
- [ ] Visualization plots created
- [ ] Parameters ranked by impact (HIGH/MEDIUM/LOW)
- [ ] Effect sizes calculated (QA change per parameter unit)

### Bayesian Optimization ✅
- [ ] Optimization completes without errors
- [ ] Best parameters found with QA score > 0.5
- [ ] Convergence plots show improvement over iterations
- [ ] Results better than or equal to random baseline

### Performance ✅
- [ ] Sensitivity analysis completes in < 4 hours
- [ ] Bayesian optimization completes in < 8 hours
- [ ] Results demonstrate parameter space understanding

## Next Steps After Testing

### If Tests Pass ✅
1. **Scale up**: Try with more subjects and iterations
2. **Compare methods**: Run full grid search for comparison
3. **Production use**: Apply to real optimization problems
4. **Merge to main**: Feature is ready for production

### If Issues Found 🔧
1. **Debug**: Use `--verbose` flag for detailed logs
2. **Simplify**: Test with fewer parameters/subjects
3. **Report**: Document issues for refinement
4. **Fallback**: Use traditional sweep methods

## Quick Test Commands (5-10 minutes)

For the fastest possible validation:

```bash
# Ultra-quick sensitivity test (just 2 parameters)
opticonn sensitivity \
  -i data/fib_samples \
  -o quick_test/sens \
  --config configs/sweep_nano.json \
  --parameters tract_count fa_threshold

# Ultra-quick Bayesian test (5 iterations)
opticonn bayesian \
  -i data/fib_samples \
  -o quick_test/bayes \
  --config configs/sweep_nano.json \
  --n-iterations 5 \
  --parameters tract_count fa_threshold
```

## Configuration Notes

- **sweep_nano.json**: Minimal config for testing (2 combinations)
- **sweep_micro.json**: Slightly larger test (more combinations)
- **braingraph_default_config.json**: Full production config

Start with `sweep_nano.json` for all tests, then scale up as needed.

---

**Ready to test?** Run the installation script, prepare your `.fib` files, and start with Phase 1! 🚀