# Configuration Guide 📋

The braingraph pipeline uses JSON configuration files to control processing parameters. Here's a guide to the available configurations.

## 🎯 Configuration Files Overview

| File | Purpose | Use Case |
|------|---------|----------|
| `01_working_config.json` | **Main config** | Regular connectivity analysis |
| `production_config.json` | **Production ready** | Batch processing, publications |
| `sweep_config.json` | **Parameter optimization** | Finding optimal tracking parameters |

## 🔧 Quick Start

### Basic Usage
```bash
# Use default working config
./01_extract_connectome.sh data/ results/

# Use specific config
./01_extract_connectome.sh --config production_config.json data/ results/
```

### Parameter Optimization
```bash
# Run parameter sweep
./01_extract_connectome.sh --sweep --config sweep_config.json data/ sweep_results/

# Quick parameter test
./01_extract_connectome.sh --quick-sweep --config sweep_config.json pilot_data/ quick_test/
```

## 📊 Configuration Sections

### Basic Settings
```json
{
  "dsi_studio_cmd": "/path/to/dsi_studio",
  "tract_count": 100000,
  "thread_count": 8
}
```

### Atlas Selection
```json
{
  "atlases": [
    "FreeSurferDKT_Cortical",
    "HCP-MMP", 
    "AAL3",
    "Schaefer400"
  ]
}
```

### Connectivity Metrics
```json
{
  "connectivity_values": [
    "count",        // Track count (most reliable)
    "fa",          // Fractional anisotropy  
    "qa",          // Quantitative anisotropy
    "ncount2",     // Normalized track count
    "mean_length"  // Average track length
  ]
}
```

### Tracking Parameters
```json
{
  "tracking_parameters": {
    "method": 0,                    // 0=Euler, 1=RK4, 2=Voxel
    "otsu_threshold": 0.6,         // Stopping threshold
    "fa_threshold": 0.05,          // FA termination
    "min_length": 10,              // Minimum track length (mm)
    "max_length": 200,             // Maximum track length (mm)
    "track_voxel_ratio": 2.0       // Seeds per voxel
  }
}
```

## 🎛️ Parameter Recommendations

### Conservative (High Quality)
```json
{
  "tract_count": 100000,
  "otsu_threshold": 0.6,
  "fa_threshold": 0.05,
  "min_length": 15
}
```

### Fast Processing  
```json
{
  "tract_count": 50000,
  "otsu_threshold": 0.5,
  "fa_threshold": 0.1,
  "min_length": 10
}
```

### High Resolution
```json
{
  "tract_count": 200000,
  "otsu_threshold": 0.7,
  "fa_threshold": 0.03,
  "min_length": 20
}
```

## 🔄 Parameter Sweeping

For parameter optimization, use the sweep configuration:

### Full Sweep (Comprehensive)
```json
{
  "sweep_parameters": {
    "otsu_range": [0.3, 0.4, 0.5, 0.6, 0.7],
    "min_length_range": [5, 10, 15, 20, 30],
    "fa_threshold_range": [0.05, 0.1, 0.15, 0.2],
    "sweep_tract_count": 50000
  }
}
```

### Quick Sweep (Fast Testing)
```json
{
  "quick_sweep": {
    "otsu_range": [0.4, 0.5, 0.6],
    "min_length_range": [10, 20],
    "fa_threshold_range": [0.05, 0.1],
    "sweep_tract_count": 25000
  }
}
```

## 🎯 Atlas-Specific Recommendations

### Cortical Analysis
```json
{
  "atlases": [
    "FreeSurferDKT_Cortical",
    "HCP-MMP", 
    "Schaefer400"
  ],
  "connectivity_values": ["count", "fa", "qa"]
}
```

### Subcortical Analysis  
```json
{
  "atlases": [
    "FreeSurferDKT_Subcortical",
    "AAL3"
  ],
  "connectivity_values": ["count", "ncount2", "mean_length"]
}
```

### Comprehensive Analysis
```json
{
  "atlases": [
    "FreeSurferDKT_Cortical",
    "FreeSurferDKT_Subcortical", 
    "HCP-MMP",
    "AAL3",
    "Schaefer400"
  ],
  "connectivity_values": ["count", "fa", "qa", "ncount2"]
}
```

## 💡 Tips and Best Practices

### Performance Optimization
- **Thread Count**: Set to number of CPU cores minus 1
- **Tract Count**: Start with 50K for testing, use 100K+ for final analysis
- **Atlas Selection**: Fewer atlases = faster processing

### Quality vs Speed Trade-offs
- **High Quality**: `tract_count: 200000`, `otsu_threshold: 0.7`
- **Balanced**: `tract_count: 100000`, `otsu_threshold: 0.6` (recommended)
- **Fast**: `tract_count: 50000`, `otsu_threshold: 0.5`

### Memory Considerations
- Large atlases (Schaefer1000) require more RAM
- Reduce `tract_count` if running out of memory
- Process fewer atlases simultaneously

## 🔍 Validation

Test your configuration before large batch runs:

```bash
# Validate configuration
python validate_setup.py --config your_config.json

# Test on pilot data
./01_extract_connectome.sh --pilot --config your_config.json data/ test_output/

# Check results
python run_pipeline.py --input test_output/organized_matrices/ --output validation_analysis/
```

## 🆘 Troubleshooting

### Common Issues

**DSI Studio not found:**
```json
{
  "dsi_studio_cmd": "/full/path/to/dsi_studio"
}
```

**Out of memory:**
```json
{
  "tract_count": 25000,
  "thread_count": 4
}
```

**Sparse matrices:**
```json
{
  "tracking_parameters": {
    "otsu_threshold": 0.4,
    "fa_threshold": 0.03
  }
}
```

---

For more help, see the main README.md or run with `--help` flags.
