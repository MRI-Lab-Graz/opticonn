# OptiConn: Brain Connectivity Optimization Pipeline

A comprehensive Python-based framework for optimizing Diffusion Spectrum Imaging (DSI) tractography parameters through cross-validated bootstrap analysis.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MRI-Lab-Graz/braingraph-pipeline.git
cd braingraph-pipeline

# Install dependencies
./install.sh

# Verify installation
python opticonn.py validate
```

### Basic Usage

```bash
# Run parameter sweep
python opticonn.py sweep \
  -i /path/to/data \
  -o /path/to/output \
  --extraction-config configs/sweep_ultra_minimal.json \
  --subjects 10

# Review results
python opticonn.py review -o /path/to/output/optimize

# Apply to full dataset
python opticonn.py apply \
  -i /path/to/full_data \
  --optimal-config /path/to/output/optimize/selected_candidate.json \
  -o /path/to/final_output
```

## Documentation

### For Users
- **[PIPELINE.md](PIPELINE.md)** - Complete pipeline documentation
  - Architecture overview
  - Detailed workflow
  - Parameter specifications
  - Quality metrics
  - Troubleshooting guide

- **[STATUS_REPORT.md](STATUS_REPORT.md)** - Project status and verification
  - Verification results
  - Performance metrics
  - Documentation checklist
  - Next steps

### For Developers
- **[SCRIPT_INVENTORY.md](SCRIPT_INVENTORY.md)** - Script analysis
  - Active vs. inactive scripts
  - Redundancy analysis
  - Consolidation recommendations
  - Usage statistics

## Features

### Core Functionality
- **Parameter Sweep**: Test parameter combinations systematically
- **Bootstrap Validation**: Two-wave cross-validation
- **Quality Assessment**: Automated QA scoring
- **Pareto Analysis**: Multi-objective optimization
- **DSI Studio Integration**: Native tractography support

### Advanced Features
- **Bayesian Optimization**: Intelligent parameter sampling
- **Sensitivity Analysis**: Parameter importance ranking
- **Multi-Atlas Support**: Test multiple atlas definitions
- **Customizable Metrics**: Define quality criteria
- **Parallel Processing**: Multi-threaded computation

## Requirements

### Hardware
- **Minimum**: 8 GB RAM, 50 GB disk space
- **Recommended**: 16+ GB RAM, 100+ GB SSD
- **Optimal**: 32+ GB RAM for large datasets

### Software
- Python 3.10+
- DSI Studio (separate installation)
- pandas, numpy, scikit-learn, dipy

### Data
- DSI Studio format files (.fz, .fib)
- Brain atlas definition files (JSON)
- Configuration files (JSON)

## Architecture

```
opticonn.py (entry point)
    ↓
scripts/opticonn_hub.py (command dispatcher)
    ├→ sweep command
    │   └→ cross_validation_bootstrap_optimizer.py
    │       ├→ extract_connectivity_matrices.py (DSI Studio)
    │       ├→ aggregate_network_measures.py
    │       ├→ optimal_selection.py
    │       └→ quick_quality_check.py
    ├→ apply command
    │   └→ subcommands/apply.py
    ├→ review command
    │   └→ subcommands/review.py
    └→ validate command
        └→ utils/validate_setup.py
```

## Active Scripts (15 Essential)

| Script | Purpose |
|--------|---------|
| `opticonn.py` | CLI launcher with venv activation |
| `opticonn_hub.py` | Command dispatcher |
| `cross_validation_bootstrap_optimizer.py` | Core sweep implementation |
| `extract_connectivity_matrices.py` | DSI Studio integration |
| `optimal_selection.py` | Parameter selection logic |
| `aggregate_network_measures.py` | Results aggregation |
| `quick_quality_check.py` | QA assessment |
| `pareto_view.py` | Pareto front analysis |
| `find_optimal_parameters.py` | Bayesian/sweep entry |
| `apply.py` | Apply parameters to dataset |
| `review.py` | Display results |
| `runtime.py` | Platform utilities |
| `validate_setup.py` | Environment validation |
| `json_validator.py` | Config validation |
| `sweep_utils.py` | Parameter utilities |

## Example Configurations

### Ultra-Minimal (Quick Test)
```json
{
  "atlases": ["FreeSurferDKT_Cortical"],
  "connectivity_values": ["count"],
  "sweep_parameters": {
    "tract_count_range": [1000, 5000],
    "sampling": {"method": "grid"}
  }
}
```

### Production (Comprehensive)
- Multiple atlases
- Multiple connectivity metrics
- Wider parameter ranges
- Random/LHS sampling
- More bootstrap iterations

See `configs/` for examples.

## Workflow

### 1. Setup & Validation
```bash
python opticonn.py validate --config configs/sweep_ultra_minimal.json
```

### 2. Parameter Sweep
```bash
python opticonn.py sweep \
  -i /data/subjects \
  -o /results \
  --extraction-config configs/sweep_ultra_minimal.json \
  --subjects 10
```

**Process**:
- Generates 2 bootstrap waves
- Samples 10 subjects per wave
- Tests parameter combinations
- Computes quality metrics
- Selects optimal parameters
- Generates Pareto analysis

### 3. Review Results
```bash
python opticonn.py review -o /results/sweep-UUID/optimize
```

**Displays**:
- Top parameter combinations
- Quality scores
- Wave comparison
- Recommendations

### 4. Apply to Full Dataset
```bash
python opticonn.py apply \
  -i /data/all_subjects \
  --optimal-config /results/sweep-UUID/optimize/selected_candidate.json \
  -o /final_results
```

## Output Structure

```
output/sweep-UUID/optimize/
├── bootstrap_qa_wave_1/
│   ├── 00_pipeline/          (processing logs)
│   ├── 01_connectivity/      (connectivity matrices)
│   └── 03_selection/         (selected parameters)
├── bootstrap_qa_wave_2/      (validation wave)
├── optimization_results/     (aggregated analysis)
├── selected_candidate.json   (best parameters)
└── sweep_results.json        (metadata)
```

## Quality Metrics

OptiConn evaluates parameters using:
- Network density
- Clustering coefficient
- Average shortest path
- Global efficiency
- Local efficiency
- Small-world index
- Modularity
- Reproducibility across bootstrap resamples

## Performance

### Single Subject, 2 Parameters
- **Runtime**: ~30-40 seconds
- **Memory**: 2 GB peak
- **Disk**: ~500 MB

### Scalability (Estimates)
- 10 subjects, 5 params: ~8 min
- 20 subjects, 10 params: ~30 min
- 50 subjects, 20 params: ~2+ hours

## Scientific Background

### Why Parameter Optimization?
- Tractography is highly sensitive to parameters
- Different parameters → different connectivity patterns
- Poor parameters → artifacts and instability
- Optimization ensures robust, reproducible results

### Why Bootstrap Validation?
- Two waves ensure generalization
- Tests reproducibility of findings
- Prevents overfitting to specific subjects
- Provides confidence intervals

### Why Quality Scoring?
- Not all parameter sets are equally valid
- Network topology matters (small-world properties)
- Reproducibility across resamples indicates stability
- Multiple metrics provide balanced assessment

## Contributing

Contributions welcome! Areas of interest:
- Additional atlases
- New quality metrics
- GPU optimization
- Visualization improvements
- Statistical extensions

## Support

### Documentation
- [PIPELINE.md](PIPELINE.md) - Complete technical guide
- [SCRIPT_INVENTORY.md](SCRIPT_INVENTORY.md) - Script analysis
- [STATUS_REPORT.md](STATUS_REPORT.md) - Development status

### Issues
- Check [TROUBLESHOOTING](#troubleshooting) section
- Review existing issues on GitHub
- Contact: karl.koschutnig@uni-graz.at

## Troubleshooting

### DSI Studio Not Found
```bash
export DSI_STUDIO_PATH=/path/to/dsi_studio
python opticonn.py sweep ...
```

### Out of Memory
```bash
python opticonn.py sweep ... --subjects 5 --max-parallel 1
```

### Configuration Errors
```bash
python opticonn.py validate --config configs/your_config.json --suggest-fixes
```

### Connectivity Matrices Invalid
- Check input .fz files
- Verify tract count parameters
- Review DSI Studio output logs

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

## License

See [LICENSE](LICENSE) file for details.

## Changelog

### Version 1.0 (Current)
- ✅ Core sweep functionality
- ✅ Bootstrap validation
- ✅ Quality assessment
- ✅ Pareto analysis
- ✅ Comprehensive documentation
- ✅ Code cleanup and simplification

## Status

**Current Status**: Production Ready ✅

- All core features implemented
- Comprehensive documentation
- Tested on real data
- Ready for scientific publication

## Project Information

- **Organization**: MRI Lab, University of Graz
- **Maintainer**: Karl Koschutnig (karl.koschutnig@uni-graz.at)
- **Repository**: https://github.com/MRI-Lab-Graz/braingraph-pipeline
- **Last Updated**: October 2025

---

**For detailed technical documentation, see [PIPELINE.md](PIPELINE.md)**
