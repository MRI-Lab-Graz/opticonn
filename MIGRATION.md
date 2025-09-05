# Pipeline Migration Guide 🔄

## What Changed?

The braingraph pipeline has been **simplified and streamlined**! We've removed the complex bash scripts (02-04) and replaced them with a single, powerful Python runner.

## ✅ Benefits of the New Pipeline

- **Simpler**: One command instead of multiple bash scripts
- **More Reliable**: Better error handling and logging
- **More Flexible**: Run individual steps or the full pipeline
- **Auto-Detection**: Automatically finds input from previous steps
- **Enhanced Features**: CSV conversion, better validation, cross-platform support

## 🔄 Migration Guide

### Old Workflow ❌
```bash
./01_extract_connectome.sh data/ results/
./02_compute_graph_metrics.sh --input pipeline_metadata.json
./03_optimize_metrics.sh metrics.csv  
./04_balanced_optimizer.sh optimization_results/
```

### New Workflow ✅
```bash
./01_extract_connectome.sh data/ results/
python run_pipeline.py --input results/organized_matrices/ --output analysis/
```

That's it! The new pipeline automatically runs steps 02-04.

## 📋 Command Comparison

| Old Commands | New Command | Notes |
|-------------|-------------|-------|
| `./02_compute_graph_metrics.sh` | `python run_pipeline.py --step 02` | Or included in full pipeline |
| `./03_optimize_metrics.sh` | `python run_pipeline.py --step 03` | Enhanced quality assessment |
| `./04_balanced_optimizer.sh` | `python run_pipeline.py --step 04` | Statistical analysis |
| All steps | `python run_pipeline.py` | **Recommended: Run all steps** |

## 🚀 Quick Migration Steps

1. **Remove old scripts** (already done automatically):
   ```bash
   # These files have been removed:
   # 02_compute_graph_metrics.sh
   # 03_optimize_metrics.sh  
   # 03_balanced_optimizer.sh
   # 04_balanced_optimizer.sh
   ```

2. **Use the new runner**:
   ```bash
   # Instead of multiple scripts, use one command:
   python run_pipeline.py --input your_matrices/ --output analysis/
   ```

3. **Update your documentation/scripts**:
   - Replace references to bash scripts 02-04 with `run_pipeline.py`
   - Update any automation scripts you have

## 🎯 Advanced Usage

### Running Individual Steps
```bash
# Step 02 only: Network optimization
python run_pipeline.py --step 02 --input matrices/ --output results/

# Step 03 only: Quality selection  
python run_pipeline.py --step 03 --input optimization_results/ --output selection/

# Step 04 only: Statistical analysis
python run_pipeline.py --step 04 --input selected_combinations/ --output stats/
```

### Auto-Detection
```bash
# Auto-detect input from step 01 output
python run_pipeline.py --output analysis/
# Will automatically find organized_matrices/ directory
```

### Verbose Logging
```bash
# Get detailed logs for debugging
python run_pipeline.py --verbose --input matrices/ --output analysis/
```

## 🔧 Configuration

The same configuration files work with the new pipeline:
- `01_working_config.json` - Main configuration
- All tracking parameters remain the same
- Sweep parameters still work with `./01_extract_connectome.sh --sweep`

## 🆘 Troubleshooting

### "Command not found" errors
Make sure you're using `python run_pipeline.py` instead of the old bash scripts.

### Input directory not found
The new pipeline auto-detects input directories. If it fails:
```bash
python run_pipeline.py --input /full/path/to/organized_matrices/ --output analysis/
```

### Want the old behavior?
The Python modules (`metric_optimizer.py`, `optimal_selection.py`, `statistical_analysis.py`) can still be run individually if needed.

## 📚 Documentation Updates

- **README.md**: Updated with new workflow
- **Examples**: All updated to use `run_pipeline.py`
- **Step 01**: Updated to show correct next steps

## ✨ Enhanced Features

The new pipeline includes all the enhanced features from the "Kopie" version:
- ✅ **CSV Conversion**: Automatic MATLAB to CSV conversion
- ✅ **Better Error Handling**: Graceful handling of missing dependencies  
- ✅ **Enhanced Logging**: Detailed progress and debug information
- ✅ **Cross-Platform**: Works on any system
- ✅ **Auto-Detection**: Smart input directory detection

---

**The new pipeline is faster, more reliable, and easier to use! 🎉**
