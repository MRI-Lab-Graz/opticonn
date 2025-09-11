#!/bin/bash

# 00_install.sh - Braingraph Pipeline Environment Setup
# Author: Braingraph Pipeline Team
# Description: Creates virtual environment and installs all required packages for the braingraph pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                                    ║"
echo "║   🧠 BRAINGRAPH PIPELINE ENVIRONMENT SETUP                                         ║"
echo "║                                                                                    ║"
echo "║   Setting up Python environment and installing all required packages              ║"
echo "║                                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv is not installed. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    echo -e "${GREEN}✅ uv installed successfully${NC}"
else
    echo -e "${GREEN}✅ uv is already installed${NC}"
fi

# Remove existing virtual environment if it exists
if [ -d "braingraph_pipeline" ]; then
    echo -e "${YELLOW}🗑️  Removing existing virtual environment...${NC}"
    rm -rf braingraph_pipeline
fi

# Create virtual environment
echo -e "${BLUE}📦 Creating virtual environment 'braingraph_pipeline'...${NC}"
uv venv braingraph_pipeline --python 3.10

# Activate the virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source braingraph_pipeline/bin/activate

# Install core Python packages
echo -e "${BLUE}📚 Installing core Python packages...${NC}"
uv pip install \
    "numpy>=1.20.0,<2.0.0" \
    "pandas>=1.5.0" \
    "matplotlib>=3.5.0" \
    "seaborn>=0.11.0" \
    "scipy>=1.8.0" \
    "scikit-learn>=1.2.0" \
    "tqdm>=4.60.0" \
    "jsonschema>=4.0.0" \
    "pathlib2>=2.3.0" \
    "typing-extensions>=4.0.0"

# Install statistical analysis packages
echo -e "${BLUE}📊 Installing statistical analysis packages...${NC}"
uv pip install \
    "statsmodels>=0.13.0" \
    "pingouin>=0.5.0" \
    "scikit-posthocs>=0.7.0"

# Install graph/network analysis packages
echo -e "${BLUE}🕸️ Installing graph and network analysis packages...${NC}"
uv pip install \
    "networkx>=2.8" \
    "igraph>=0.10.0" \
    "bctpy>=0.5.2" \
    "python-louvain>=0.16"

# Install neuroimaging packages
echo -e "${BLUE}🧠 Installing neuroimaging packages...${NC}"
uv pip install \
    "nilearn>=0.10.0" \
    "nibabel>=4.0.0" \
    "dipy>=1.5.0"

# Install additional analysis packages
echo -e "${BLUE}📈 Installing additional analysis packages...${NC}"
uv pip install \
    "plotly>=5.0.0" \
    "dash>=2.0.0" \
    "jupyter>=1.0.0" \
    "ipywidgets>=8.0.0" \
    "openpyxl>=3.0.0" \
    "xlsxwriter>=3.0.0" \
    "h5py>=3.0.0"

echo ""
echo -e "${GREEN}✅ Package installation completed successfully!${NC}"
echo ""
echo -e "${BLUE}🎯 Environment Summary:${NC}"
echo "• Virtual environment: braingraph_pipeline/"
echo "• Python version: 3.10"
echo "• All required packages installed"
echo ""
echo -e "${YELLOW}📋 To activate the environment:${NC}"
echo "  source braingraph_pipeline/bin/activate"
echo ""
echo -e "${YELLOW}📋 To deactivate the environment:${NC}"
echo "  deactivate"
echo ""
echo -e "${GREEN}🚀 Environment ready for braingraph pipeline!${NC}"
