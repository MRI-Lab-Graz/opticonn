# Installation Guide

This directory contains platform-specific installation scripts for setting up the Braingraph Pipeline environment.

## 📋 Available Installation Scripts

### Unix/Linux/macOS
- **`00_install_new.sh`** (RECOMMENDED) - Latest version with all dependencies
- **`00_install.sh`** (LEGACY) - Original version, missing some packages

### Windows
- **`00_install_windows.bat`** - Windows batch script with same functionality

## 🔍 Key Differences

### `00_install_new.sh` vs `00_install.sh`

The **NEW** version includes one additional package:
```bash
# OLD version (00_install.sh)
"python-louvain>=0.16"

# NEW version (00_install_new.sh) 
"python-louvain>=0.16" \
"community>=1.0.0"
```

The `community>=1.0.0` package provides additional community detection algorithms that may be useful for network analysis.

## 🚀 Quick Start

### Unix/Linux/macOS
```bash
# Use the recommended new version
./00_install_new.sh
source braingraph_pipeline/bin/activate
```

### Windows
```cmd
REM Run the Windows batch file
00_install_windows.bat
REM Then activate the environment
braingraph_pipeline\Scripts\activate.bat
```

## 📦 What Gets Installed

All scripts install the same core packages:

### Core Python Packages
- numpy, pandas, matplotlib, seaborn
- scipy, scikit-learn, tqdm
- jsonschema, pathlib2, typing-extensions

### Statistical Analysis
- statsmodels, pingouin, scikit-posthocs

### Graph/Network Analysis
- networkx, igraph, bctpy
- python-louvain, community (NEW version only)

### Neuroimaging
- nilearn, nibabel, dipy

### Additional Tools
- plotly, dash, jupyter
- openpyxl, xlsxwriter, h5py

## 🛠️ Prerequisites

### All Platforms
- Python 3.10 or higher
- Internet connection for package downloads

### Automatic Installation
- **uv package manager** (automatically installed if missing)

### Manual Prerequisites (if needed)
- Git (for cloning repository)
- C++ compiler (for some neuroimaging packages)

## 🔧 Troubleshooting

### Common Issues

**Permission errors (Unix/macOS):**
```bash
chmod +x 00_install_new.sh
./00_install_new.sh
```

**Python not found (Windows):**
- Install Python from https://www.python.org/downloads/
- Ensure "Add Python to PATH" is checked during installation

**uv installation fails:**
- Check internet connection
- Try manual installation from https://docs.astral.sh/uv/

**Package compilation errors:**
- Install build tools for your platform
- Consider using conda instead of pip for problematic packages

## ✅ Validation

After installation, validate your setup:

```bash
# Activate environment
source braingraph_pipeline/bin/activate  # Unix/macOS
# OR
braingraph_pipeline\Scripts\activate.bat  # Windows

# Validate installation
python validate_setup.py --config 01_working_config.json

# Test basic functionality
python run_pipeline.py --test-config test_extraction_only.json
```

## 🔄 Updating

To update your environment:

```bash
# Remove old environment
rm -rf braingraph_pipeline  # Unix/macOS
# OR
rmdir /s braingraph_pipeline  # Windows

# Run installation script again
./00_install_new.sh  # Unix/macOS
# OR
00_install_windows.bat  # Windows
```

## 💡 Recommendation

**Use `00_install_new.sh`** for Unix/macOS systems as it includes the most recent package set with all community detection algorithms. The legacy `00_install.sh` can be removed unless you have specific compatibility requirements.
