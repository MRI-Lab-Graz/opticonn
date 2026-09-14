# OptiConn v2.0 - User-Friendly Edition

## 🎯 What's New

I've created a completely redesigned OptiConn package in the `/opticonn` folder that makes brain connectivity parameter optimization much more user-friendly. Here's what changed:

### 🔄 Two-Phase Workflow

**PHASE 1: SWEEP** - Find optimal parameters
- Test parameter combinations on 3-5 subjects (not all!)
- Get top 3 candidates in 15-30 minutes
- Bootstrap validation for reliability

**PHASE 2: APPLY** - Use optimal parameters  
- Apply best settings to all subjects
- Generate analysis-ready datasets
- Quality reports and visualizations

### 🚀 Simple Commands

```bash
# Find optimal parameters (Phase 1)
python opticonn.py sweep --data /path/to/subjects

# Apply to all subjects (Phase 2)  
python opticonn.py apply --config results/best_params.json --data /path/to/subjects

# Or do everything at once
python opticonn.py auto --data /path/to/subjects
```

## 📁 New Structure

```
opticonn/
├── opticonn.py                    # Main user-friendly interface
├── install.sh                     # Easy setup script
├── README.md                      # Comprehensive documentation
├── configs/
│   ├── quick_sweep.json           # 2-5 minutes (learning)
│   ├── default_sweep.json         # 15-30 minutes (standard)
│   └── comprehensive_sweep.json   # 1-2 hours (research)
└── utils/
    ├── config_helper.py           # Create/validate configurations
    └── results_viewer.py          # Analyze results
```

## 🔧 Key Improvements

### 1. **User-Friendly Interface**
- Clear command structure: `sweep`, `apply`, `auto`
- Helpful error messages and progress indicators
- Automatic environment validation
- `--dry-run` mode to preview actions

### 2. **Smart Configuration**
- Pre-built configs for different use cases
- Automatic parameter grid estimation
- Configuration validation and warnings
- Helper scripts for custom configs

### 3. **Better Documentation**
- Step-by-step quick start guide
- Troubleshooting section
- Example workflows for different scenarios
- Clear explanations of results

### 4. **Quality Assurance**
- Environment validation (`python opticonn.py validate`)
- Bootstrap cross-validation built-in
- Pareto front analysis and visualization
- Comprehensive logging

## 🎓 Learning Path

### 1. **Beginner** (5 minutes)
```bash
./install.sh
python opticonn.py validate
python opticonn.py sweep --data /data/subjects --quick --subjects 2
```

### 2. **Standard Use** (30 minutes)
```bash
python opticonn.py auto --data /data/subjects
```

### 3. **Research Grade** (2 hours)
```bash
python opticonn.py auto --config configs/comprehensive_sweep.json --data /data/subjects --subjects 5
```

## 🔗 Dependencies

OptiConn provides a self-contained user-friendly interface. Vendored pipeline
scripts live under `opticonn/scripts/` so the repository can be operated
independently of the original upstream project. The structure is:

```
opticonn/
├── opticonn.py                    # Main user-friendly interface
├── install.sh                     # Easy setup script (creates .venv)
├── README.md                      # Comprehensive documentation
├── configs/                       # Sweep configuration presets
└── scripts/                       # Vendored pipeline scripts (importable as `scripts`)
```

## 💡 Benefits

1. **Faster Learning Curve**: New users can be productive in minutes
2. **Cleaner Workflow**: Two clear phases instead of complex scripts
3. **Better Quality Control**: Built-in validation and cross-checking  
4. **Flexible Configuration**: Easy to customize for different needs
5. **Comprehensive Documentation**: Everything needed to get started

## 🚀 Next Steps

1. **Test the install script**: `./install.sh`
2. **Validate environment**: `python opticonn.py validate`
3. **Try quick test**: `python opticonn.py sweep --quick --data /test/data`
4. **Read the README**: Complete documentation with examples
5. **Use utilities**: `utils/config_helper.py` and `utils/results_viewer.py`

The new OptiConn is designed to make brain connectivity analysis accessible to researchers without requiring deep technical knowledge of the underlying optimization algorithms, while still providing the flexibility for advanced users to customize everything they need.

---

**Ready to optimize brain connectivity the easy way!** 🧠✨