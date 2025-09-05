# Simplified Braingraph Pipeline Workflow

## Overview

The pipeline has been streamlined to use **JSON configuration files** instead of complex command-line parsing:

### ✅ **Current Approach (Recommended)**
```bash
# 1. Connectivity extraction with JSON config
python extract_connectivity_matrices.py --config 01_working_config.json --batch raw_data/ output/

# 2. Full analysis pipeline  
python run_pipeline.py --step analysis --input output/organized_matrices/ --config production_config.json
```

### ❌ **Old Approach (Deprecated)**
```bash
# Complex bash script with many parameters
./01_extract_connectome.sh --atlases "AAL3,HCP-MMP" --metrics "count,fa,qa" --tracks 100000 raw_data/ output/
```

## 🎯 **Recommended Workflow**

### **Step 1: Connectivity Extraction**
```bash
# Single subject
python extract_connectivity_matrices.py --config 01_working_config.json subject.fz output/

# Batch processing
python extract_connectivity_matrices.py --config 01_working_config.json --batch raw_data/ output/

# Pilot test (recommended first)
python extract_connectivity_matrices.py --config 01_working_config.json --batch --pilot raw_data/ output/
```

### **Step 2: Complete Analysis Pipeline**
```bash
# Run analysis steps 02-04
python run_pipeline.py --step analysis --input output/organized_matrices/ --config production_config.json

# Or run everything from raw data (if you prefer single command)
python run_pipeline.py --step all --data-dir raw_data/ --extraction-config 01_working_config.json --config production_config.json
```

## 📋 **Configuration Files**

| File | Purpose | Use Case |
|------|---------|----------|
| `01_working_config.json` | Connectivity extraction | Routine analysis |
| `sweep_config.json` | Parameter sweeping | Research/optimization |
| `production_config.json` | Analysis steps 02-04 | Final analysis |

## 🔧 **Configuration Benefits**

1. **Reproducible**: Save exact parameters used
2. **Shareable**: Easy to share configs between team members  
3. **Versioned**: Track parameter changes in git
4. **Flexible**: Easy to create configs for different studies
5. **Clean**: No complex command-line parsing

## 📁 **File Structure After Processing**

```
output/
├── organized_matrices/          # Step 01 output
│   ├── by_atlas/
│   ├── by_metric/
│   ├── combined/
│   └── logs/
├── optimization_results/        # Step 02 output
├── selected_combinations/       # Step 03 output  
└── statistical_results/         # Step 04 output
```

## 🚀 **Quick Start**

```bash
# 1. Validate your setup
python validate_setup.py --config 01_working_config.json

# 2. Test with pilot
python extract_connectivity_matrices.py --config 01_working_config.json --batch --pilot raw_data/ output/

# 3. Run full analysis
python run_pipeline.py --step analysis --input output/organized_matrices/ --config production_config.json
```

This approach eliminates command-line complexity while maintaining full flexibility through JSON configuration files.
