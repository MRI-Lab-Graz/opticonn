# Braingraph Pipeline 🧠

A streamlined pipeline for brain connectivity analysis using DSI Studio, optimized for neuroscience research.

## 🎯 Quick Start

### Prerequisites
```bash
# Install the pipeline
./00_install.sh

# Validate setup (recommended)
python validate_setup.py --config 01_working_config.json
```

### Basic Workflow

```bash
# Step 1: Extract connectivity matrices
./01_extract_connectome.sh data/ results/

# Steps 2-4: Run analysis pipeline
python run_pipeline.py --input results/organized_matrices/ --output analysis/
```

That's it! The pipeline will automatically:
- ✅ Optimize network quality metrics
- ✅ Select best atlas/metric combinations  
- ✅ Run statistical analysis
- ✅ Generate publication-ready results

## 📁 Pipeline Overview

### Core Components

| Component | Purpose | Input | Output |
|-----------|---------|--------|--------|
| **01_extract_connectome.sh** | Extract connectivity matrices | `.fz` files | Organized matrices |
| **run_pipeline.py** | Analysis pipeline (steps 2-4) | Organized matrices | Statistical results |

### Python Modules

| Module | Purpose |
|--------|---------|
| `extract_connectivity_matrices.py` | DSI Studio interface with enhanced CSV support |
| `metric_optimizer.py` | Network quality optimization |
| `optimal_selection.py` | Quality-based atlas/metric selection |
| `statistical_analysis.py` | Statistical modeling and analysis |

## 🚀 Usage Examples

### Single Subject
```bash
# Extract and analyze single subject
./01_extract_connectome.sh subject001.fz ./results/
python run_pipeline.py --input results/organized_matrices/ --output analysis/
```

### Batch Processing
```bash
# Process multiple subjects
./01_extract_connectome.sh --batch /data/subjects/ ./results/
python run_pipeline.py --input results/organized_matrices/ --output analysis/
```

### Parameter Optimization
```bash
# Run parameter sweep for optimal tracking settings
./01_extract_connectome.sh --sweep /data/subjects/ ./sweep_results/
```

### Custom Analysis
```bash
# Run individual steps
python run_pipeline.py --step 02 --input matrices/ --output opt/     # Optimization only
python run_pipeline.py --step 03 --input opt/ --output selection/    # Selection only  
python run_pipeline.py --step 04 --input selection/ --output stats/  # Statistics only
```

## 📊 Output Structure

```
results/
├── connectivity_matrices/           # Raw DSI Studio output
├── organized_matrices/             # Clean, organized structure
│   ├── {atlas}/                   # By atlas
│   │   └── {metric}/              # By connectivity metric
│   └── by_subject/                # By subject
└── analysis/                      # Analysis results
    ├── optimization_results/      # Step 2: Quality metrics
    ├── selected_combinations/     # Step 3: Best combinations  
    └── statistical_results/       # Step 4: Statistical analysis
```

## ⚙️ Configuration

### Main Config: `01_working_config.json`
```json
{
  "dsi_studio_cmd": "/path/to/dsi_studio",
  "atlases": ["AAL3", "HCP-MMP", "Schaefer400"],
  "connectivity_values": ["count", "fa", "qa", "ncount2"],
  "tract_count": 100000,
  "tracking_parameters": {
    "otsu_threshold": 0.6,
    "fa_threshold": 0.05,
    "min_length": 10
  }
}
```

### Parameter Sweeps
For tracking parameter optimization, the config includes `sweep_parameters`:
```json
{
  "sweep_parameters": {
    "otsu_range": [0.3, 0.4, 0.5],
    "min_length_range": [10, 20, 30],
    "fa_threshold_range": [0.05, 0.1, 0.15]
  }
}
```

## 🧪 Advanced Features

### Quality-First Methodology
The pipeline implements a pure quality assessment approach:
- **Network topology metrics** (sparsity, small-worldness, modularity)
- **No statistical bias** in quality assessment
- **Reproducible selection** of optimal combinations

### Enhanced Data Formats
- **Automatic CSV conversion** from MATLAB formats
- **Robust error handling** for missing dependencies
- **Cross-platform compatibility**

### Parameter Optimization
- **Intelligent parameter sweeps** for tracking optimization
- **Sparsity-based quality metrics** 
- **Automated best parameter detection**

## 🔧 Development

### Adding New Atlases
Edit `01_working_config.json`:
```json
{
  "atlases": ["NewAtlas", "ExistingAtlas1", "ExistingAtlas2"]
}
```

### Custom Metrics
Add to connectivity values:
```json
{
  "connectivity_values": ["count", "fa", "custom_metric"]
}
```

### Extending Analysis
The modular design allows easy extension:
- Add new optimization criteria in `metric_optimizer.py`
- Extend selection logic in `optimal_selection.py`
- Add statistical models in `statistical_analysis.py`

## 📚 Documentation

- **DSI Studio Tools**: `dsi_studio_tools/README.md`
- **Parameter Guide**: `dsi_studio_tools/TRACK_COUNT_GUIDE.md`
- **Batch Processing**: `dsi_studio_tools/BATCH_PROCESSING_GUIDE.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample data
5. Submit a pull request

## 📄 License

See `LICENSE` file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Run validation: `python validate_setup.py`
3. Review log files in `logs/` directory
4. Open an issue on GitHub

---

**Built for neuroscience research with ❤️**
