# OptiConn: Brain Connectivity Optimization Pipeline

## Overview

OptiConn is a comprehensive Python-based pipeline for optimizing **Diffusion Spectrum Imaging (DSI)** tractography parameters through cross-validated bootstrap analysis. It systematically evaluates parameter combinations to identify optimal settings that produce robust and reproducible connectivity matrices.

### Key Features

- **Bayesian Optimization**: Intelligently samples parameter space to find optimalsettings with fewer evaluations
- **Parameter Sweep**: Exhaustive or sampled grid search across parameter combinations
- **Cross-Validated Bootstrap Analysis**: Two-wave validation strategy to ensure parameter robustness
- **Quality Assessment**: Automated QA scoring based on network properties and stability
- **Pareto Front Analysis**: Identifies trade-offs between different quality metrics
- **DSI Studio Integration**: Native integration with DSI Studio for tractography processing

## Architecture

### Entry Point: `opticonn.py`

The main command-line interface that:
1. Locates and activates the virtual environment
2. Delegates to `scripts/opticonn_hub.py` for command handling

```bash
python opticonn.py [command] [options]
```

### Core Pipeline Scripts

#### 1. **`scripts/opticonn_hub.py`** - Main Hub
Main command dispatcher supporting:
- `sweep`: Parameter sweep with bootstrap validation
- `apply`: Apply optimized parameters to full dataset
- `review`: Display and analyze optimization results
- Configuration validation and setup checks

**Subcommands:**
- `scripts/subcommands/find_optimal_parameters.py`: Bayesian or sweep optimization
- `scripts/subcommands/apply.py`: Apply parameters to dataset
- `scripts/subcommands/review.py`: Display optimization results

#### 2. **`scripts/cross_validation_bootstrap_optimizer.py`** - Bootstrap Optimizer
Runs the main optimization workflow:
1. Generates two bootstrap waves (train/validation)
2. Randomly samples subjects for each wave
3. Runs tractography with parameter combinations
4. Aggregates connectivity matrices
5. Computes quality scores
6. Selects optimal parameters

**Workflow:**
```
Wave 1 (Optimization):
  - Sample N subjects
  - Run M parameter combinations
  - Bootstrap resample B times
  - Compute quality metrics
  - Select best parameters

Wave 2 (Validation):
  - Use different subjects
  - Run best parameters from Wave 1
  - Validate parameter stability
  - Generate final recommendations
```

#### 3. **`scripts/extract_connectivity_matrices.py`** - DSI Studio Wrapper
Manages DSI Studio tractography execution:
- Converts subject data to DSI Studio format
- Executes fiber tracking with specified parameters
- Extracts connectivity matrices
- Handles error recovery and logging

**Key Functions:**
- `run_dsi_studio()`: Execute DSI Studio command
- `extract_network_properties()`: Compute connectivity statistics
- `validate_results()`: Quality checks on output

#### 4. **`scripts/aggregate_network_measures.py`** - Results Aggregation
Collects and aggregates metrics across runs:
- Reads individual subject results
- Computes network statistics (density, clustering, diameter, etc.)
- Generates aggregated output CSV
- Handles missing data gracefully

**Output:** `aggregated_network_measures.csv`

#### 5. **`scripts/optimal_selection.py`** - Parameter Selection
Selects best parameter combinations based on:
- Quality scores (QA metrics)
- Stability across waves
- Network property consistency
- User-specified thresholds

**Selection Criteria:**
- Minimum quality score threshold
- Recommendation from bootstrap analysis
- Top N combinations overall
- Top N per atlas

#### 6. **`scripts/bootstrap_qa_validator.py`** - QA Validation
Validates connectivity matrix quality using bootstrap resampling:
- Evaluates robustness of connectivity patterns
- Computes reproducibility scores
- Generates diagnostic reports

#### 7. **`scripts/quick_quality_check.py`** - Quick QA
Fast quality assessment of results:
- Checks for outliers
- Validates connectivity ranges
- Reports potential issues
- Generates warning levels

#### 8. **`scripts/pareto_view.py`** - Pareto Front Analysis
Analyzes trade-offs between optimization objectives:
- Identifies non-dominated solutions
- Generates Pareto front visualization
- Recommends balanced parameter sets

### Utility Scripts

#### `scripts/utils/runtime.py`
Platform-specific runtime utilities:
- `configure_stdio()`: Configure output streams
- `propagate_no_emoji()`: Environment setup for subprocesses
- Path handling for Windows long paths

#### `scripts/utils/validate_setup.py`
Validates pipeline environment:
- Checks DSI Studio installation
- Verifies Python dependencies
- Validates input data
- Checks disk space

#### `scripts/json_validator.py`
Configuration validation:
- Validates JSON syntax
- Checks required parameters
- Suggests fixes for common errors
- Type checking for parameter values

#### `scripts/sweep_utils.py`
Parameter sweep utilities:
- `build_param_grid_from_config()`: Generate parameter combinations
- `grid_product()`: Full grid sampling
- `random_sampling()`: Random parameter sampling
- `lhs_sampling()`: Latin Hypercube Sampling

### Configuration Files

Located in `configs/`:

#### `braingraph_default_config.json`
Default configuration template with:
- Atlases to use
- Connectivity metrics
- Tractography parameters
- QA thresholds
- Bootstrap settings

**Key Sections:**
```json
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["count"],
  "tractography_parameters": {
    "fa_threshold": 0.1,
    "min_length": 10,
    "track_voxel_ratio": 2.0
  },
  "sweep_parameters": {
    "tract_count_range": [1000, 5000],
    "fa_threshold_range": [0.05, 0.3],
    "sampling": {"method": "grid"}
  },
  "bootstrap": {"n_iterations": 5},
  "qa_thresholds": {"min_quality_score": 0.5}
}
```

#### `sweep_ultra_minimal.json`
Minimal configuration for quick testing:
- Single atlas (FreeSurferDKT_Cortical)
- Single metric (count)
- 2 parameter combinations
- Fast processing

## Workflow Examples

### 1. Parameter Sweep

```bash
python opticonn.py sweep \
  -i /path/to/data \
  -o /path/to/output \
  --extraction-config configs/sweep_ultra_minimal.json \
  --subjects 10
```

**Process:**
1. Validates configuration and environment
2. Generates bootstrap wave configurations
3. Samples 10 subjects for each wave
4. Tests parameter combinations
5. Runs DSI Studio tractography
6. Extracts connectivity matrices
7. Computes quality metrics
8. Selects optimal parameters
9. Generates Pareto front analysis

**Output Directory Structure:**
```
output/sweep-UUID/
├── optimize/
│   ├── bootstrap_qa_wave_1/          # Wave 1 results
│   │   ├── 00_pipeline/              # Processing logs
│   │   ├── 01_connectivity/          # Connectivity matrices
│   │   └── 03_selection/             # Selected parameters
│   ├── bootstrap_qa_wave_2/          # Wave 2 validation
│   ├── optimization_results/         # Aggregated results
│   ├── selected_candidate.json       # Best parameters
│   └── sweep_results.json            # Optimization metadata
```

### 2. Review Results

```bash
python opticonn.py review -o /path/to/output/optimize
```

Displays:
- Selected parameter combinations
- Quality scores
- Comparison between waves
- Recommendations

### 3. Apply Parameters to Full Dataset

```bash
python opticonn.py apply \
  -i /path/to/full_data \
  --optimal-config /path/to/selected_candidate.json \
  -o /path/to/final_output
```

## Parameter Space

### Tractography Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `tract_count` | 1,000-200,000 | Number of fiber tracts to track |
| `fa_threshold` | 0.05-0.3 | Fractional anisotropy threshold |
| `min_length` | 5-50 mm | Minimum fiber tract length |
| `turning_angle` | 30-90° | Maximum tract turning angle |
| `step_size` | 0.5-2.0 mm | Integration step size |
| `track_voxel_ratio` | 1-5 | Tracks per voxel |
| `connectivity_threshold` | 0.0001-0.01 | Connection weight threshold |

### Quality Metrics

| Metric | Description |
|--------|-------------|
| Network Density | Proportion of possible connections |
| Clustering Coefficient | Local network clustering |
| Average Shortest Path | Network efficiency |
| Small World Index | Presence of small-world topology |
| Modularity | Community structure strength |
| Efficiency (Global/Local) | Information flow efficiency |

## Data Format

### Input Data
- **Format**: `.fz` files (DSI Studio format)
- **Structure**: One file per subject
- **Location**: Organized in directory with subject naming

### Output Data
- **Connectivity Matrices**: CSV format, NxN adjacency matrices
- **Network Measures**: Aggregated statistics CSV
- **Quality Scores**: JSON format
- **Optimization Metadata**: JSON with parameter information

## Bootstrap Analysis Details

### Wave Structure
1. **Wave 1 (Optimization)**
   - Sample N subjects randomly
   - Test all parameter combinations
   - Bootstrap resample connectivity matrices (B iterations)
   - Compute average quality metrics
   - Rank parameters by quality

2. **Wave 2 (Validation)**
   - Use different N subjects (no overlap with Wave 1)
   - Run top-ranked parameters from Wave 1
   - Bootstrap resample connectivity matrices (B iterations)
   - Compare quality metrics between waves
   - Validate parameter selection stability

### Why Bootstrap?
- Accounts for sampling variability
- Provides confidence intervals on metrics
- Tests robustness of connectivity patterns
- Identifies unstable parameter regions

## Quality Assessment Pipeline

### Automatic QA Scoring

1. **Network Validity Checks**
   - Non-empty connectivity matrices
   - Reasonable density values
   - Positive edge weights

2. **Statistical Checks**
   - Outlier detection (connectivity extremes)
   - Distribution shape analysis
   - Reproducibility across resamples

3. **Network Property Analysis**
   - Small-world properties
   - Efficiency metrics
   - Community structure

4. **Cross-Wave Stability**
   - Metric correlation between waves
   - Selection agreement
   - Parameter robustness score

### Quality Score Formula

```
Quality = (stability_score × 0.4) + (network_validity × 0.3) + 
          (reproducibility × 0.2) + (consistency × 0.1)
```

## Computational Considerations

### Runtime Estimates

| Component | Time |
|-----------|------|
| DSI Studio tractography (per subject) | 5-30 min |
| Connectivity extraction | 1-5 min |
| QA analysis | 2-10 min |
| Full sweep (10 subjects, 10 parameters) | 3-5 hours |
| Bayesian optimization (50 iterations) | 6-12 hours |

### Memory Requirements
- **Minimum**: 8 GB RAM
- **Recommended**: 16+ GB RAM
- **High-resolution images**: 32+ GB RAM

### Parallel Processing
- Adjustable via `--max-parallel` flag
- Parallelizes subject processing
- Thread pool management built-in

## Advanced Features

### Bayesian Optimization
Uses scikit-optimize for intelligent parameter sampling:
- Gaussian Process surrogate model
- Expected Improvement acquisition function
- Significantly fewer evaluations than grid search

```bash
python opticonn.py optimize --method bayesian \
  --n-iterations 50 \
  -i /path/to/data \
  -o /path/to/output \
  --config configs/braingraph_default_config.json
```

### Sensitivity Analysis
Analyzes parameter sensitivity:
- Importance ranking
- Parameter interaction effects
- Robust parameter regions

### Pareto Front Optimization
Multi-objective optimization considering:
- Quality vs. computational cost
- Different atlas selections
- Trade-offs between metrics

## Troubleshooting

### Common Issues

**Issue**: DSI Studio not found
- **Solution**: Set `DSI_STUDIO_PATH` environment variable
- **Example**: `export DSI_STUDIO_PATH=/path/to/dsi_studio`

**Issue**: Configuration validation fails
- **Solution**: Run with `--suggest-fixes` flag
- **Example**: `opticonn validate --config config.json --suggest-fixes`

**Issue**: Out of memory
- **Solution**: Reduce `--max-parallel` or `--subjects`
- **Example**: `opticonn sweep --subjects 5 --max-parallel 1`

**Issue**: Connectivity matrices invalid
- **Solution**: Check tractography parameters
- **Solution**: Verify input data quality
- **Solution**: Review DSI Studio output

## Scientific Background

### DSI vs. DTI
DSI provides higher angular resolution for crossing fibers:
- 515 diffusion-weighted directions (vs. ~30 for DTI)
- Better resolution of complex fiber geometries
- More robust fiber tracking

### Network Neuroscience
OptiConn produces metrics relevant to:
- Structural connectivity analysis
- Network topology studies
- Graph-theoretic brain analysis
- Connectomics research

### Bootstrap Validation Rationale
Two-wave approach ensures:
- Parameters generalize to unseen data
- Metrics are stable across sampling
- Results are reproducible
- Overfitting is minimized

## Citation

If you use OptiConn in your research, please cite:

```bibtex
@software{braingraph_pipeline,
  title={OptiConn: Brain Connectivity Optimization Pipeline},
  author={Koschutnig, Karl and [Contributors]},
  year={2025},
  url={https://github.com/MRI-Lab-Graz/braingraph-pipeline}
}
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional atlases support
- More QA metrics
- GPU acceleration for DSI Studio
- Advanced statistical methods
- Visualization improvements

## References

Key publications underlying the methods:

1. **Connectivity Matrix Extraction**
   - Yeh, F. C., et al. (2016). Quantifying differences and similarities in whole-brain white matter architecture using diffusion spectrum imaging.

2. **Bootstrap Validation**
   - Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.

3. **Network Analysis**
   - Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity.

4. **Optimization Methods**
   - Bergstra, J., et al. (2013). Hyperopt: A hyperparameter optimization library.

## License

See LICENSE file for details.

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Maintainer**: MRI Lab, University of Graz
