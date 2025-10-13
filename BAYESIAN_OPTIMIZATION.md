# Bayesian Optimization for Tractography Parameters

## Overview

Bayesian optimization is a smart, efficient approach to finding optimal tractography parameters. Instead of testing all possible combinations (grid search), it learns from each evaluation and focuses on the most promising parameter regions.

**Key advantages:**
- 🚀 **Much faster**: Finds optimal parameters in 20-50 evaluations instead of 100s
- 🧠 **Intelligent**: Learns which parameter combinations work best
- 📊 **Adaptive**: Focuses search on promising regions
- 🎯 **Efficient**: Handles continuous and discrete parameters elegantly

## How It Works

### Traditional Grid Search
```
Test ALL combinations: 5^7 = 78,125 combinations
Time: Days to weeks
```

### Bayesian Optimization
```
1. Start with a few random samples (5)
2. Build a model of parameter → QA relationship
3. Pick next parameters that are likely to improve QA
4. Update model and repeat
5. Converge to optimal in ~30 iterations

Time: Hours to days
```

## Quick Start

### 1. Run Bayesian Optimization

```bash
opticonn bayesian \
  -i data/fib_samples \
  -o results/bayesian_opt \
  --config configs/base_config.json \
  --n-iterations 30
```

### 2. Review Results

```bash
# Check progress
cat results/bayesian_opt/bayesian_optimization_progress.json

# View best parameters
cat results/bayesian_opt/bayesian_optimization_results.json
```

### 3. Apply Best Parameters

The Bayesian optimizer saves the best configuration found. You can apply it to your full dataset:

```bash
# Extract best config from results
python -c "
import json
with open('results/bayesian_opt/bayesian_optimization_results.json') as f:
    data = json.load(f)
    best_params = data['best_parameters']
    print(json.dumps(best_params, indent=2))
"
```

## Command Options

```bash
opticonn bayesian [OPTIONS]

Required:
  -i, --data-dir PATH          Input data directory (.fz/.fib.gz files)
  -o, --output-dir PATH        Output directory for results
  --config PATH                Base configuration JSON file

Optional:
  --n-iterations INT           Number of optimization iterations (default: 30)
  --n-bootstrap INT            Bootstrap samples per evaluation (default: 3)
  --verbose                    Show detailed progress
  --no-emoji                   Disable emoji output
```

## Parameter Space

The optimizer searches over these parameters by default:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| tract_count | 10,000 - 200,000 | 100,000 | Number of streamlines |
| fa_threshold | 0.05 - 0.3 | 0.1 | FA termination threshold |
| min_length | 5 - 50 mm | 10 | Minimum tract length |
| turning_angle | 30 - 90° | 60 | Maximum turning angle |
| step_size | 0.5 - 2.0 mm | 1.0 | Integration step size |
| track_voxel_ratio | 1.0 - 5.0 | 2.0 | Seeds per voxel |
| connectivity_threshold | 0.0001 - 0.01 | 0.001 | Connection threshold |

## Understanding Results

### Output Files

```
results/bayesian_opt/
├── bayesian_optimization_results.json    # Final results & best parameters
├── bayesian_optimization_progress.json   # Progress tracking
├── iterations/                           # Individual iteration results
│   ├── iteration_0001_config.json
│   ├── iteration_0001/
│   │   ├── 01_connectivity/
│   │   ├── 02_optimization/
│   │   └── 03_selection/
│   ├── iteration_0002_config.json
│   └── ...
```

### Results JSON Structure

```json
{
  "optimization_method": "bayesian",
  "n_iterations": 30,
  "best_qa_score": 0.8765,
  "best_parameters": {
    "tract_count": 125000,
    "fa_threshold": 0.12,
    "min_length": 15,
    "turning_angle": 55.0,
    "step_size": 1.2,
    "track_voxel_ratio": 2.5,
    "connectivity_threshold": 0.0008
  },
  "all_iterations": [
    {
      "iteration": 1,
      "qa_score": 0.65,
      "params": {...}
    },
    ...
  ]
}
```

## Advanced Usage

### Custom Parameter Ranges

Modify `scripts/bayesian_optimizer.py` to customize parameter ranges:

```python
@dataclass
class ParameterSpace:
    # Customize these ranges
    tract_count: Tuple[int, int] = (50000, 300000)
    fa_threshold: Tuple[float, float] = (0.08, 0.25)
    # ... more parameters
```

### Resume Interrupted Optimization

The optimizer saves progress after each iteration. To resume:

```bash
# Check last completed iteration
cat results/bayesian_opt/bayesian_optimization_progress.json | grep completed_iterations

# Continue from where it left off (feature coming soon)
```

## When to Use Bayesian Optimization

### ✅ **Good for:**
- **Large parameter spaces**: When grid search would take too long
- **Expensive evaluations**: Each tractography run takes significant time
- **Continuous parameters**: Naturally handles continuous ranges
- **Unknown relationships**: Discovers complex parameter interactions
- **Limited compute budget**: Get good results with few evaluations

### ❌ **Consider alternatives:**
- **Very small spaces**: Grid search might be faster for 2-3 parameters
- **Well-understood parameters**: If you know good values already
- **Need exhaustive coverage**: When you must test all combinations
- **Debugging**: Sensitivity analysis better for understanding impacts

## Comparison with Other Methods

| Method | Evaluations | Time | Coverage | Intelligence |
|--------|-------------|------|----------|--------------|
| **Grid Search** | 100-10,000+ | Days-Weeks | Complete | None |
| **Random Search** | 50-200 | Hours-Days | Random | None |
| **Bayesian Opt** | **20-50** | **Hours-Days** | **Smart** | **High** |
| **Sensitivity** | 10-20 | Hours | Gradient | Medium |

## Tips & Best Practices

### 1. **Start Small**
```bash
# Test with 10 iterations first
opticonn bayesian ... --n-iterations 10
```

### 2. **Use Good Baseline**
Start with a reasonable baseline configuration that you know works.

### 3. **Monitor Progress**
```bash
# Watch progress in real-time
tail -f results/bayesian_opt/bayesian_optimization_progress.json
```

### 4. **Combine with Sensitivity Analysis**
```bash
# First: Understand which parameters matter
opticonn sensitivity -i data -o sens --config base.json

# Then: Focus Bayesian optimization on important parameters
opticonn bayesian -i data -o bayes --config base.json --n-iterations 30
```

### 5. **Validate Results**
Always validate the best parameters with cross-validation:

```bash
# Run bootstrap validation on best params
opticonn sweep -i data -o validation \
  --config best_params.json --quick
```

## Technical Details

### Gaussian Process Model

Bayesian optimization uses a Gaussian Process (GP) to model the relationship:

```
QA_score = f(tract_count, fa_threshold, min_length, ...)
```

The GP provides:
- **Mean prediction**: Expected QA score for untested parameters
- **Uncertainty**: How confident we are about the prediction
- **Acquisition function**: Balance exploration vs exploitation

### Acquisition Function

The optimizer uses Expected Improvement (EI):
```
EI(x) = E[max(f(x) - f(x_best), 0)]
```

This balances:
- **Exploitation**: Test parameters likely to improve QA
- **Exploration**: Test uncertain regions to learn more

### Convergence

The optimizer typically converges in 3 phases:

1. **Exploration (iter 1-10)**: Tests diverse parameter combinations
2. **Refinement (iter 11-25)**: Focuses on promising regions
3. **Exploitation (iter 26-30)**: Fine-tunes best parameters

## Troubleshooting

### Problem: Optimization gets stuck

**Solution**: Increase random initialization
```python
# In bayesian_optimizer.py
result = gp_minimize(
    ...
    n_random_starts=10  # Increase from 5 to 10
)
```

### Problem: Takes too long per iteration

**Solution**: Use faster bootstrap evaluation
```bash
opticonn bayesian ... --n-bootstrap 2  # Reduce from 3
```

### Problem: Best parameters seem unrealistic

**Solution**: Adjust parameter ranges in `ParameterSpace`

## Future Enhancements

- [ ] Multi-objective optimization (QA + computation time)
- [ ] Parallel evaluations (test multiple parameters simultaneously)
- [ ] Resume from checkpoint
- [ ] Custom acquisition functions
- [ ] Automated parameter range discovery

## References

- Snoek et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms"
- Shahriari et al. (2016). "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
- [scikit-optimize documentation](https://scikit-optimize.github.io/)

## Support

For questions or issues:
1. Check the [main README](README.md)
2. Review [SENSITIVITY_ANALYSIS.md](SENSITIVITY_ANALYSIS.md) for complementary approach
3. Open an issue on GitHub
