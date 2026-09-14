# OptiConn - User-Friendly Brain Connectivity Parameter Optimization

OptiConn simplifies brain connectivity analysis by automatically finding optimal DSI Studio parameters for your dataset. Instead of guessing parameters, OptiConn tests different combinations and tells you which work best.

## 🎯 What OptiConn Does

**Phase 1: Find Best Parameters** 
- Tests parameter combinations on 3-5 subjects
- Identifies top 3 candidates based on quality metrics  
- Takes 15-30 minutes instead of weeks of manual testing

**Phase 2: Apply to Full Dataset**
- Uses optimal parameters on all subjects
- Produces analysis-ready connectivity matrices
- Generates quality reports and visualizations

## 🚀 Quick Start

### 1. Installation (Requires DSI Studio Path)
```bash
# Clone the repository and install into a local virtualenv
git clone https://github.com/your-org/opticonn.git

# Navigate to opticonn and install (this creates/uses ./opticonn/.venv)
cd opticonn
DSI Studio must be installed locally (GUI bundle or standalone binary). Have the absolute path to the `dsi_studio` executable ready, e.g.:

macOS (App bundle):
  /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
Linux (custom install):
  /opt/dsi_studio/dsi_studio

Run the installer. You must provide or export the path explicitly (interactive prompting removed in favor of a required flag):
```bash
# Option A: Flag (recommended)
./install.sh --dsi-studio /Applications/dsi_studio.app/Contents/MacOS/dsi_studio

# Option B: Environment variable
DSI_STUDIO_CMD=/Applications/dsi_studio.app/Contents/MacOS/dsi_studio ./install.sh
```
```

### 2. Activate Environment (Required for every session)
```bash
# Option 1: Use the activation script (RECOMMENDED)
source activate.sh

# Option 2: Activate the created local virtualenv directly
source .venv/bin/activate
# DSI_STUDIO_CMD must point to a valid executable (enforced at install time).
# You can override temporarily in a session by re-exporting the variable.
```

### 3. Test Your Setup
```bash
python opticonn.py validate
```

### 4. Find Optimal Parameters (Phase 1)
```bash
python opticonn.py sweep --data /path/to/subjects --quick
```

### 5. Apply to All Subjects (Phase 2)  
```bash
python opticonn.py apply --config results/best_parameters.json --data /path/to/subjects
```

### 6. Or Run Everything at Once
```bash
python opticonn.py auto --data /path/to/subjects
```

## 📊 Example Workflows

### Quick Testing (2-5 minutes)
```bash
# Test with minimal parameters - great for learning
python opticonn.py sweep --data /data/subjects --quick --subjects 2
```

### Standard Analysis (15-30 minutes)
```bash
# Balanced optimization for most research
python opticonn.py auto --config configs/default_sweep.json --data /data/subjects
```

### Research-Grade Analysis (1-2 hours)
```bash
# Comprehensive optimization for publications
python opticonn.py auto --config configs/comprehensive_sweep.json --data /data/subjects --subjects 5
```

## 📁 Configuration Files

| File | Use Case | Runtime | Parameters |
|------|----------|---------|------------|
| `configs/quick_sweep.json` | Learning/Testing | 2-5 min | Minimal grid |
| `configs/default_sweep.json` | Standard research | 15-30 min | Balanced ranges |
| `configs/comprehensive_sweep.json` | Publications | 1-2 hours | Extensive grid |

### DSI Studio Configuration

OptiConn requires an explicit DSI Studio path. The installer enforces this via the `--dsi-studio` flag (or `DSI_STUDIO_CMD` env var). You may still embed a path in configs for documentation or alternate versions, but the environment variable / flag path is authoritative at install time:

```json
{
  "dsi_studio_cmd": "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
  "atlases": ["FreeSurferDKT_Cortical"],
  ...
}
```

**Benefits of embedding in config:**
- ✅ Version-controlled DSI Studio path used for that analysis
- ✅ Easier reproducibility and multi-version testing
- ✅ Fallback discovery if the environment variable is not set prior to runtime

If both are provided, the environment variable `DSI_STUDIO_CMD` takes precedence at runtime.

### Custom Configuration
Create your own sweep config by modifying parameter ranges:
```json
{
  "dsi_studio_cmd": "/path/to/your/dsi_studio",
  "sweep_parameters": {
    "fa_threshold_range": [0.05, 0.10, 0.15, 0.20],
    "min_length_range": [10, 15, 20, 30],
    "tract_count_range": [250000, 500000, 1000000]
  }
}
```

## 🎛️ Command Reference

### Phase 1: Parameter Sweep
```bash
python opticonn.py sweep [options]
```
**Options:**
- `--config CONFIG` - Sweep configuration file
- `--data DATA_DIR` - Directory with .fz/.fib.gz files  
- `--output OUTPUT` - Results directory
- `--subjects N` - Number of subjects for testing (default: 3)
- `--quick` - Use minimal parameter grid
- `--verbose` - Detailed output
- `--dry-run` - Show commands without running

### Phase 2: Apply Parameters
```bash
python opticonn.py apply [options]
```
**Options:**
- `--config CONFIG` - Optimal parameters file (from Phase 1)
- `--data DATA_DIR` - Directory with .fz/.fib.gz files
- `--output OUTPUT` - Results directory  
- `--candidate N` - Which candidate to use (1=best, 2=second, etc.)

### Complete Workflow
```bash
python opticonn.py auto [options]
```
**Options:**
- Combines all options from sweep and apply
- Automatically runs Phase 1 → Phase 2

### Environment Check
```bash
python opticonn.py validate
```

## 📈 Understanding Results

### Phase 1 Output
```
results/
├── optimize/
│   └── optimization_results/
│       ├── top3_candidates.json      # Best parameter combinations
│       ├── pareto_front.csv          # All tested combinations
│       └── pareto_front.png          # Visualization
└── logs/
    └── opticonn_YYYYMMDD_HHMMSS.log  # Detailed logs
```

### Phase 2 Output  
```
analysis/
├── selected/
│   └── 03_selection/
│       ├── FreeSurferSeg_analysis_ready.csv       # Ready for statistics
│       ├── FreeSurferDKT_Cortical_analysis_ready.csv
│       └── optimal_selection_summary.txt
└── logs/
```

### Top Candidates Format
```json
[
  {
    "atlas": "FreeSurferDKT_Cortical",
    "connectivity_metric": "fa", 
    "average_score": 0.85,
    "parameters": {
      "fa_threshold": 0.15,
      "min_length": 20,
      "tract_count": 500000
    }
  }
]
```

### Troubleshooting

### Environment Issues
```bash
# Check what's wrong
python opticonn.py validate

# Common fixes - ALWAYS activate environment first:
source activate.sh
# OR activate the local venv directly:
source .venv/bin/activate
```

### No Subject Files Found
```bash
# OptiConn looks for .fz and .fib.gz files
ls /path/to/data/*.fz
ls /path/to/data/*.fib.gz

# Make sure files are in the data directory, not subdirectories
```

### DSI Studio Issues
```bash
# Test DSI Studio manually (use path from your config file)
"/Applications/dsi_studio.app/Contents/MacOS/dsi_studio" --help

# Find DSI Studio location
find /Applications -name "dsi_studio*" 2>/dev/null
find /usr/local -name "dsi_studio*" 2>/dev/null

# Update your config file if DSI Studio moved:
# Edit configs/default_sweep.json and update "dsi_studio_cmd" path
```

### Memory Issues
```bash
# Reduce parameters for large datasets
python opticonn.py sweep --config configs/quick_sweep.json --subjects 2

# Or modify tract_count in config:
"tract_count_range": [100000, 250000]  # Instead of millions
```

## 🧪 Examples Directory

Check `examples/` for:
- `quick_start.sh` - Complete workflow example
- Sample configurations  
- Expected directory structure

## 🔬 How It Works

1. **Parameter Generation**: Creates combinations from your ranges
2. **Quality Testing**: Runs each combination on subset of subjects
3. **Metric Scoring**: Evaluates sparsity, modularity, efficiency
4. **Bootstrap Validation**: Confirms stability across samples
5. **Selection**: Ranks combinations and exports top candidates
6. **Full Analysis**: Applies best parameters to complete dataset

## 📚 More Information

- **Brain connectivity basics**: See DSI Studio documentation
- **Parameter meanings**: Check `configs/` file comments
-- **Advanced usage**: Modify the vendored `scripts/` inside this repository (`opticonn/scripts/`)
- **Publication methods**: Use comprehensive_sweep.json + bootstrap validation

## 💡 Tips

- **Start small**: Use `--quick` and `--subjects 2` for first runs
- **Monitor progress**: Use `--verbose` to see detailed progress  
- **Save configs**: Keep successful parameter files for future use
- **Batch processing**: Run multiple datasets with same optimal parameters
- **Quality check**: Always review the pareto_front.png visualization

## 🤝 Support

- Check logs in `results/logs/` for detailed error information
- Use `--dry-run` to preview commands before execution
- Run `validate` command to check environment setup
-- See the `docs/` directory or `opticonn/scripts/` for advanced troubleshooting

---

**OptiConn v2.0** - Making brain connectivity analysis accessible and reproducible.