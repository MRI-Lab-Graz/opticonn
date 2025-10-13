# Sensitivity Analysis for Tractography Parameters

## Overview

Sensitivity analysis helps you understand **which tractography parameters have the most impact** on network quality scores. Instead of blindly optimizing all parameters, this approach identifies the "high-leverage" parameters that deserve your attention.

**Key benefits:**
- 📊 **Understand impact**: See which parameters matter most for YOUR data
- 🎯 **Focus efforts**: Optimize important parameters, fix less important ones
- 💡 **Gain insights**: Learn parameter-QA relationships
- ⚡ **Save time**: Avoid optimizing parameters that don't matter

## How It Works

### The Gradient Concept

For each parameter, we compute the sensitivity (gradient):

```
Sensitivity = ∂QA / ∂parameter
```

This tells us: "How much does QA change when this parameter changes?"

### Process

1. **Establish baseline**: Run tractography with your baseline configuration
2. **Perturb each parameter**: Change one parameter by 10% (keeping others fixed)
3. **Measure impact**: Compare QA scores
4. **Calculate sensitivity**: Compute gradient (change in QA / change in parameter)
5. **Rank parameters**: Sort by absolute impact

## Quick Start

### 1. Run Sensitivity Analysis

```bash
opticonn sensitivity \
  -i data/fib_samples \
  -o results/sensitivity \
  --config configs/base_config.json
```

### 2. Review Results

```bash
# View text results
cat results/sensitivity/sensitivity_analysis_results.json

# View visualization
open results/sensitivity/sensitivity_analysis_plot.png
```

### 3. Interpret Findings

Example output:
```
Parameters ranked by impact:

1. fa_threshold            🔴 HIGH
   Sensitivity: -0.085123 QA per 0.01 FA
   Baseline: 0.1000
   
2. tract_count             🟡 MEDIUM  
   Sensitivity: +0.012456 QA per 10k tracks
   Baseline: 100000
   
3. step_size               🟢 LOW
   Sensitivity: +0.000823 QA per 1mm
   Baseline: 1.0
```

**Interpretation**:
- `fa_threshold` has **HIGH** impact → **Optimize this carefully**
- `tract_count` has **MEDIUM** impact → **Consider optimizing**
- `step_size` has **LOW** impact → **Can use default value**

## Command Options

```bash
opticonn sensitivity [OPTIONS]

Required:
  -i, --data-dir PATH          Input data directory (.fz/.fib.gz files)
  -o, --output-dir PATH        Output directory for results
  --config PATH                Baseline configuration JSON file

Optional:
  --parameters PARAM [PARAM ...]  Specific parameters to analyze
  --perturbation FLOAT         Perturbation factor (default: 0.1 = 10%)
  --verbose                    Show detailed progress
  --no-emoji                   Disable emoji output
```

## Parameters Analyzed

By default, analyzes all tractography parameters:

| Parameter | Typical Range | Impact on QA |
|-----------|---------------|--------------|
| tract_count | 10k - 200k | Network sampling density |
| fa_threshold | 0.05 - 0.3 | Tract termination, false positives |
| min_length | 5 - 50 mm | Short spurious tracts |
| turning_angle | 30 - 90° | Tract curvature, branching |
| step_size | 0.5 - 2.0 mm | Integration accuracy |
| track_voxel_ratio | 1.0 - 5.0 | Seeding density |
| connectivity_threshold | 0.0001 - 0.01 | Network sparsity |

## Understanding Results

### Output Files

```
results/sensitivity/
├── sensitivity_analysis_results.json    # Numerical results
├── sensitivity_analysis_plot.png        # Visual summary
├── config_baseline.json                 # Baseline config used
├── config_perturbed_fa_threshold.json   # Example perturbed config
├── baseline/                            # Baseline results
│   ├── 01_connectivity/
│   ├── 02_optimization/
│   └── 03_selection/
├── perturbed_fa_threshold/              # Perturbed results
└── ...
```

### Results JSON Structure

```json
{
  "baseline_qa": 0.7234,
  "perturbation_factor": 0.1,
  "parameters": {
    "fa_threshold": {
      "sensitivity": -0.085123,
      "baseline_value": 0.1,
      "perturbed_value": 0.11,
      "baseline_qa": 0.7234,
      "abs_sensitivity": 0.085123
    },
    "tract_count": {
      "sensitivity": 0.012456,
      "baseline_value": 100000,
      "perturbed_value": 110000,
      "baseline_qa": 0.7234,
      "abs_sensitivity": 0.012456
    }
  },
  "ranked_parameters": [
    "fa_threshold",
    "tract_count",
    ...
  ]
}
```

### Visualization

The plot shows two views:

1. **Sensitivity with sign** (left):
   - Green bars = Increasing parameter increases QA
   - Red bars = Increasing parameter decreases QA

2. **Absolute sensitivity** (right):
   - Shows magnitude of impact regardless of direction
   - Helps identify which parameters to prioritize

## Advanced Usage

### Analyze Specific Parameters

```bash
# Only analyze the most important parameters
opticonn sensitivity \
  -i data -o results \
  --config base.json \
  --parameters tract_count fa_threshold min_length
```

### Adjust Perturbation Size

```bash
# Use 20% perturbation instead of 10%
opticonn sensitivity \
  -i data -o results \
  --config base.json \
  --perturbation 0.2
```

### Sequential Analysis

```bash
# 1. First pass with large perturbation (20%)
opticonn sensitivity ... --perturbation 0.2

# 2. Identify high-impact parameters

# 3. Second pass with small perturbation (5%) on important parameters
opticonn sensitivity ... --perturbation 0.05 \
  --parameters fa_threshold tract_count
```

## Interpreting Sensitivity Values

### Magnitude

| Abs(Sensitivity) | Impact Level | Action |
|------------------|--------------|--------|
| > 0.01 | 🔴 **HIGH** | **Must optimize carefully** |
| 0.001 - 0.01 | 🟡 **MEDIUM** | Consider optimizing |
| < 0.001 | 🟢 **LOW** | Can use default value |

### Sign

- **Positive (+)**: Increasing parameter → Higher QA
- **Negative (-)**: Increasing parameter → Lower QA

### Context Matters

Sensitivity depends on:
- Your specific data (tissue types, resolution, quality)
- Baseline parameter values (non-linear relationships)
- Parameter interactions (not captured by single-parameter analysis)

## Workflow Integration

### Recommended Workflow

```bash
# Step 1: Sensitivity analysis (fast, ~1-2 hours)
opticonn sensitivity -i data -o sens --config base.json

# Step 2: Review results
cat sens/sensitivity_analysis_results.json

# Step 3: Focus optimization on high-impact parameters
# Option A: Manual grid search on important params only
opticonn sweep -i data -o sweep --config focused_sweep.json

# Option B: Bayesian optimization (uses all params but weights by impact)
opticonn bayesian -i data -o bayes --config base.json --n-iterations 30

# Step 4: Validate best parameters
opticonn review -o bayes/optimize
opticonn apply -i data --optimal-config bayes/optimize/selected_candidate.json -o final
```

## Use Cases

### 1. **New Dataset** → Understand your data

```bash
# First time working with a new dataset?
# Run sensitivity analysis to understand which parameters matter

opticonn sensitivity -i new_data -o explore --config default.json
```

### 2. **Limited Compute** → Focus optimization

```bash
# Have limited computational resources?
# Identify high-impact parameters and only optimize those

# 1. Sensitivity analysis
opticonn sensitivity -i data -o sens --config base.json

# 2. Review results, identify top 3 parameters
# 3. Create focused sweep config with only those 3 parameters
# 4. Run smaller sweep
```

### 3. **Research Question** → Parameter importance

```bash
# Publishing paper on tractography methods?
# Use sensitivity analysis to report which parameters matter most

opticonn sensitivity -i data -o sens --config base.json
# → Report in Methods section
```

### 4. **Quality Control** → Verify assumptions

```bash
# Think certain parameters don't matter?
# Verify with sensitivity analysis

opticonn sensitivity -i data -o sens --config base.json \
  --parameters step_size smoothing
```

## Limitations

### What Sensitivity Analysis DOES:
- ✅ Identifies important parameters
- ✅ Measures first-order effects (gradients)
- ✅ Provides guidance for optimization
- ✅ Fast and interpretable

### What it DOESN'T:
- ❌ Find optimal parameter values (use Bayesian opt for that)
- ❌ Capture complex interactions between parameters
- ❌ Account for non-linear relationships fully
- ❌ Replace full parameter sweep validation

### Complementary Approaches

Sensitivity analysis works best combined with:

1. **Bayesian Optimization**: Sensitivity → identify important params → Bayesian → find optimal values
2. **Grid Search**: Sensitivity → focus grid → faster grid search
3. **Expert Knowledge**: Sensitivity → validate assumptions → adjust priors

## Technical Details

### Gradient Estimation

Uses finite differences:

```
∂QA/∂p ≈ (QA(p + Δp) - QA(p)) / Δp
```

Where:
- `p` = parameter value
- `Δp` = perturbation (default: 10% of baseline)
- `QA(p)` = quality score at parameter value p

### Normalization

Sensitivities are in original parameter units:
- `tract_count`: QA per 10,000 tracks
- `fa_threshold`: QA per 0.01 FA
- `min_length`: QA per 1 mm

This allows direct comparison of practical significance.

### Statistical Considerations

- Each evaluation includes bootstrap sampling (default: 3 samples)
- Results may vary slightly between runs
- Larger perturbations → more robust but less local
- Smaller perturbations → more local but noisier

## Troubleshooting

### Problem: Low baseline QA score

**Solution**: Check your baseline configuration first
```bash
# Validate baseline gives reasonable results
opticonn apply -i data --config base.json -o test_baseline
```

### Problem: All sensitivities near zero

**Possible causes**:
1. Perturbation too small → Increase with `--perturbation 0.2`
2. Data variability high → Increase bootstrap samples
3. QA metric not sensitive → Try different QA metric

### Problem: Inconsistent results between runs

**Solution**: Increase number of bootstrap samples
```python
# In sensitivity_analyzer.py, modify:
n_bootstrap_samples = 5  # Increase from 3
```

### Problem: Takes too long

**Solution**: Analyze fewer parameters
```bash
# Only analyze top suspects
opticonn sensitivity ... \
  --parameters tract_count fa_threshold
```

## Example Results

### Typical Finding for DWI Data

```
High-impact parameters (>0.01):
  1. fa_threshold: -0.082  ← Most critical
  2. otsu_threshold: -0.045
  
Medium-impact parameters (0.001-0.01):
  3. tract_count: +0.008
  4. turning_angle: -0.006
  5. min_length: +0.003
  
Low-impact parameters (<0.001):
  6. step_size: +0.0007   ← Can use defaults
  7. smoothing: -0.0003
```

**Recommendation**: Focus optimization on `fa_threshold` and `otsu_threshold`, use defaults for `step_size` and `smoothing`.

## Future Enhancements

- [ ] Multi-variate sensitivity (parameter interactions)
- [ ] Global sensitivity analysis (Sobol indices)
- [ ] Automated perturbation size selection
- [ ] Confidence intervals on sensitivities
- [ ] Parameter correlation analysis

## References

- Saltelli et al. (2008). "Global Sensitivity Analysis: The Primer"
- Morris (1991). "Factorial Sampling Plans for Preliminary Computational Experiments"
- Sobol (2001). "Global sensitivity indices for nonlinear mathematical models"

## Support

For questions or issues:
1. Check the [main README](README.md)
2. Review [BAYESIAN_OPTIMIZATION.md](BAYESIAN_OPTIMIZATION.md) for complementary approach
3. Open an issue on GitHub
