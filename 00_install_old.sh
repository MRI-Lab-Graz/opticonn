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
    "python-louvain>=0.16" \
    "community>=1.0.0"

# Install neuroimaging packages
echo -e "${BLUE}🧠 Installing neuroimaging packages...${NC}"
uv pip install \
    "nilearn>=0.10.0" \
    "nibabel>=4.0.0" \
    "dipy>=1.5.0"

# Install optional packages for advanced analysis
echo -e "${BLUE}📈 Installing advanced analysis packages...${NC}"
uv pip install \
    "plotly>=5.0.0" \
    "dash>=2.0.0" \
    "jupyter>=1.0.0" \
    "ipywidgets>=8.0.0" \
    "openpyxl>=3.0.0"

# Create activation wrapper scripts
echo -e "${BLUE}🔧 Creating wrapper scripts...${NC}"

# Main pipeline runner
cat > run_pipeline.sh << 'EOF'
#!/bin/bash

# Main pipeline runner with automatic environment activation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/connectivity_pipeline"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Run the main pipeline
if [ -f "$SCRIPT_DIR/pipeline.py" ]; then
    python "$SCRIPT_DIR/pipeline.py" "$@"
else
    echo "⚠️  pipeline.py not found. Running in development mode..."
    echo "📁 Current directory: $SCRIPT_DIR"
    echo "🐍 Python environment: $(which python)"
    echo "📦 Available packages:"
    pip list | grep -E "(numpy|pandas|networkx|bctpy)"
fi
EOF

chmod +x run_pipeline.sh

# Atlas sweeper runner
cat > run_atlas_sweep.sh << 'EOF'
#!/bin/bash

# Atlas sweeping tool with automatic environment activation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/connectivity_pipeline"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

if [ -f "$SCRIPT_DIR/atlas_sweep.py" ]; then
    python "$SCRIPT_DIR/atlas_sweep.py" "$@"
else
    echo "⚠️  atlas_sweep.py will be created in the pipeline setup"
fi
EOF

chmod +x run_atlas_sweep.sh

# Metric optimizer runner
cat > run_metric_optimizer.sh << 'EOF'
#!/bin/bash

# Metric optimization tool with automatic environment activation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/connectivity_pipeline"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

if [ -f "$SCRIPT_DIR/metric_optimizer.py" ]; then
    python "$SCRIPT_DIR/metric_optimizer.py" "$@"
else
    echo "⚠️  metric_optimizer.py will be created in the pipeline setup"
fi
EOF

chmod +x run_metric_optimizer.sh

# Statistical analysis runner
cat > run_analysis.sh << 'EOF'
#!/bin/bash

# Statistical analysis tool with automatic environment activation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/connectivity_pipeline"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

if [ -f "$SCRIPT_DIR/statistical_analysis.py" ]; then
    python "$SCRIPT_DIR/statistical_analysis.py" "$@"
else
    echo "⚠️  statistical_analysis.py will be created in the pipeline setup"
fi
EOF

chmod +x run_analysis.sh

# Interactive environment activation
cat > activate_env.sh << 'EOF'
#!/bin/bash

# Interactive environment activation for development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/connectivity_pipeline"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

echo "🔧 Activating connectivity_pipeline environment..."
echo "📍 Use 'deactivate' to exit the environment"
echo ""

# Start a new shell with the environment activated
bash --rcfile <(echo "source $VENV_DIR/bin/activate; PS1='(connectivity_pipeline) \u@\h:\w\$ '")
EOF

chmod +x activate_env.sh

# Create development utilities
cat > test_environment.sh << 'EOF'
#!/bin/bash

# Test the installed environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/connectivity_pipeline"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "🧪 Testing environment..."
python -c "
import sys
print(f'Python version: {sys.version}')
print()

# Test core packages
packages = ['numpy', 'pandas', 'networkx', 'matplotlib', 'seaborn', 'scipy', 'sklearn', 'statsmodels', 'tqdm']
print('Core packages:')
for pkg in packages:
    try:
        exec(f'import {pkg}')
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')

print()

# Test brain connectivity packages
brain_packages = ['bct', 'community', 'nilearn', 'nibabel']
print('Brain connectivity packages:')
for pkg in brain_packages:
    try:
        exec(f'import {pkg}')
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')

print()
print('🎉 Environment test completed!')
"
EOF

chmod +x test_environment.sh

echo ""
echo -e "${GREEN}✅ Installation completed successfully!${NC}"
echo ""
echo -e "${BLUE}🎯 Quick Start Guide:${NC}"
echo ""
echo -e "${YELLOW}1. Test your environment:${NC}"
echo "   ./test_environment.sh"
echo ""
echo -e "${YELLOW}2. Run the complete pipeline:${NC}"
echo "   ./run_pipeline.sh --help"
echo "   ./run_pipeline.sh /path/to/fz_files/ /path/to/output/"
echo ""
echo -e "${YELLOW}3. Run individual components:${NC}"
echo "   ./run_atlas_sweep.sh /path/to/fz_files/ /path/to/matrices/"
echo "   ./run_metric_optimizer.sh /path/to/matrices/ /path/to/optimized/"
echo "   ./run_analysis.sh /path/to/optimized/ /path/to/results/"
echo ""
echo -e "${YELLOW}4. Interactive development:${NC}"
echo "   ./activate_env.sh  # Start interactive session"
echo "   # Then use normal Python commands"
echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo "• The pipeline Python scripts will be created next"
echo "• Configure your analysis parameters"
echo "• Place your .fz files in the input directory"
echo "• Run the atlas sweep to explore all options"
echo "• Use metric optimization to find best parameters"
echo "• Perform your statistical analysis"
echo ""
echo -e "${GREEN}🚀 Ready to analyze brain connectivity!${NC}"
