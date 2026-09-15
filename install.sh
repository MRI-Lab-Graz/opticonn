#!/bin/bash

# OptiConn Installation Script
# Installs the user-friendly OptiConn package for brain connectivity analysis
#
# Usage:
#   ./install.sh                                          # MRtrix3 backend only
#   ./install.sh --dsi-studio /absolute/path/to/dsi_studio  # also set up DSI Studio
#   # or
#   DSI_STUDIO_CMD=/absolute/path/to/dsi_studio ./install.sh
#
# Flags:
#   --dsi-studio <path>   (Optional; enables the DSI Studio backend)
#   --non-interactive      Fail immediately if requirements not met (default behavior now)
#   --help                 Show usage and exit

set -e  # Exit on any error

NON_INTERACTIVE=1
DSI_STUDIO_FLAG=""

print_usage() {
    cat <<'USAGE'
OptiConn Installer
==================
Backends (install at least one yourself):
    MRtrix3      tckgen and tck2connectome on PATH (no flag needed)
    DSI Studio   pass its path with --dsi-studio (or DSI_STUDIO_CMD)
Optional:
    --dsi-studio /path/to/dsi_studio    Absolute path to DSI Studio executable
    --non-interactive                   (default) Fail fast on errors
    --help                              Show this help text

Examples:
    ./install.sh
    ./install.sh --dsi-studio /Applications/dsi_studio.app/Contents/MacOS/dsi_studio
    DSI_STUDIO_CMD=/opt/dsi_studio/dsi_studio ./install.sh
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dsi-studio)
            shift || { echo "Missing value for --dsi-studio" >&2; exit 1; }
            DSI_STUDIO_FLAG="$1"; shift ;;
        --dsi-studio=*)
            DSI_STUDIO_FLAG="${1#*=}"; shift ;;
        --non-interactive)
            NON_INTERACTIVE=1; shift ;;
        --help|-h)
            print_usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            print_usage
            exit 1 ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                                    ║"
echo "║   🧠 OPTICONN INSTALLATION                                                         ║"
echo "║                                                                                    ║"
echo "║   User-Friendly Brain Connectivity Parameter Optimization                         ║"
echo "║                                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in the opticonn directory
if [[ ! -f "opticonn.py" ]]; then
    echo -e "${RED}❌ Please run this script from the opticonn directory${NC}"
    echo -e "${YELLOW}💡 Expected files: opticonn.py, configs/, etc.${NC}"
    exit 1
fi

# Use a local virtual environment inside this repository (.venv)
VENV_PATH=".venv"
echo -e "${BLUE}🔧 Ensuring local virtual environment at: $VENV_PATH${NC}"

if [[ ! -d "$VENV_PATH" ]]; then
    echo -e "${YELLOW}⚠️  Local virtual environment not found — creating ${VENV_PATH}...${NC}"
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✅ Created virtual environment at $VENV_PATH${NC}"
fi

# Activate the local venv
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo -e "${GREEN}✅ Virtual environment activated: $VIRTUAL_ENV${NC}"
        echo -e "${BLUE}📍 Using Python: $(which python)${NC}"
    else
        echo -e "${RED}❌ Failed to activate virtual environment${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Virtual environment activation script not found: $VENV_PATH/bin/activate${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.12"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo -e "${GREEN}✅ Python $PYTHON_VERSION is compatible${NC}"
else
    echo -e "${RED}❌ Python 3.12+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi

# ----------------------------------------------------------------------------
# Backend: DSI Studio if a path is given, otherwise MRtrix3 only
# ----------------------------------------------------------------------------
# Priority: flag > env var
if [[ -n "$DSI_STUDIO_FLAG" ]]; then
  export DSI_STUDIO_CMD="$DSI_STUDIO_FLAG"
fi

if [[ -z "$DSI_STUDIO_CMD" ]]; then
  VALIDATE_BACKEND="mrtrix3"
  echo -e "${BLUE}ℹ️  No DSI Studio path given: setting up for the MRtrix3 backend.${NC}"
  echo "   (Re-run with --dsi-studio /path/to/dsi_studio to add DSI Studio.)"
else
  VALIDATE_BACKEND="dsi_studio"
  if [[ ! -f "$DSI_STUDIO_CMD" ]]; then
    echo -e "${RED}❌ DSI Studio executable not found: $DSI_STUDIO_CMD${NC}"
    exit 1
  fi

  echo -e "${GREEN}✅ DSI Studio: $DSI_STUDIO_CMD${NC}"
  echo -e "${BLUE}💾 Persist by adding to shell profile:${NC}"
  echo "   export DSI_STUDIO_CMD=\"$DSI_STUDIO_CMD\""

  # Save configuration for persistent environment setup
  CONFIG_FILE=".opticonn_config"
  cat > "$CONFIG_FILE" << EOF
# OptiConn Configuration File
# This file is automatically generated during installation
# Do not edit manually - re-run install.sh to update

DSI_STUDIO_CMD="$DSI_STUDIO_CMD"
EOF
  echo -e "${GREEN}✅ Configuration saved to $CONFIG_FILE${NC}"
fi

# Install Python dependencies into the local virtualenv
REQ_FILE="requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo -e "${BLUE}📦 Installing Python dependencies from $REQ_FILE into the local venv...${NC}"
    # upgrade pip first
    "$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
    "$VENV_PATH/bin/pip" install -r "$REQ_FILE"
    echo -e "${GREEN}✅ Python dependencies installed into $VENV_PATH${NC}"
else
    echo -e "${YELLOW}⚠️  No requirements.txt found; skipping pip install${NC}"
fi

# Test OptiConn
echo -e "${BLUE}🧪 Testing OptiConn installation...${NC}"

# Make sure we're using the correct Python
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo -e "${BLUE}📍 Using Python: $(which $PYTHON_CMD)${NC}"
echo -e "${BLUE}📍 Python version: $($PYTHON_CMD --version)${NC}"

if "$VENV_PATH/bin/python" opticonn.py validate --backend "$VALIDATE_BACKEND" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ OptiConn validation passed ($VALIDATE_BACKEND)${NC}"
else
    echo -e "${YELLOW}⚠️  OptiConn validation warnings (run manually to see details)${NC}"
    echo -e "${BLUE}💡 Try: python opticonn.py validate --backend $VALIDATE_BACKEND${NC}"
fi

# Create example directory structure
echo -e "${BLUE}📁 Setting up example directory structure...${NC}"

mkdir -p examples/{data,results}
mkdir -p docs

# Create a simple example script
cat > examples/quick_start.sh << 'EOF'
#!/bin/bash

# OptiConn Quick Start Example
# This script demonstrates the basic OptiConn workflow

echo "🧠 OptiConn Quick Start Example"
echo "==============================="
echo ""

# Check if data directory has sample files
if [[ ! -d "data" ]] || [[ -z "$(ls -A data)" ]]; then
    echo "❌ No data files found in examples/data/"
    echo "💡 Please add some .fz or .fib.gz files to examples/data/ first"
    echo ""
    echo "Expected structure:"
    echo "  examples/"
    echo "  ├── data/"
    echo "  │   ├── subject001.fz"
    echo "  │   ├── subject002.fz"
    echo "  │   └── ..."
    echo "  └── quick_start.sh"
    exit 1
fi

echo "📊 Found $(ls data/fib_samples/*.fz 2>/dev/null | wc -l) subject files"
echo ""

# Activate environment (use the local venv created by install.sh)
echo "🔧 Activating environment..."
if [[ -f "../.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ../.venv/bin/activate
else
    echo "⚠️  Local virtualenv not found at ../.venv — make sure to run ../install.sh"
fi

# Run OptiConn in quick mode (use local opticonn entrypoint)
echo "🚀 Running OptiConn quick test..."
python3 ../opticonn.py sweep \
    --data data/fib_samples \
    --output results \
    --quick \
    --subjects 2

echo ""
echo "✅ Quick start example completed!"
echo "📁 Check results/ directory for outputs"
EOF

chmod +x examples/quick_start.sh

# Final setup summary
echo ""
echo -e "${GREEN}✅ OptiConn installation completed!${NC}"
echo ""
echo -e "${BLUE}🎯 Quick Start (run these commands):${NC}"
echo "1. Make sure you're in the opticonn directory: cd $(pwd)"
echo "2. Activate environment: source .venv/bin/activate"
echo "3. Test installation:   python opticonn.py validate --backend $VALIDATE_BACKEND"
echo ""
echo -e "${YELLOW}📝 Important Notes:${NC}"
echo "• Always activate the virtual environment before using OptiConn"
echo "• The backend is chosen by \"backend\" in the sweep config (mrtrix3 or dsi_studio)"
echo ""
echo -e "${BLUE}📚 Available configurations:${NC}"
echo "• configs/mrtrix_quick_sweep.json  - MRtrix3 starter sweep"
echo "• configs/quick_sweep.json         - DSI Studio fast testing"
echo "• configs/default_sweep.json       - DSI Studio balanced sweep"
echo "• configs/comprehensive_sweep.json - DSI Studio extensive sweep"
echo ""
echo -e "${BLUE}📖 Usage Examples:${NC}"
echo "• Phase 1 (screen):  python opticonn.py sweep --config configs/mrtrix_quick_sweep.json --data /data"
echo "• Phase 2 (apply):   python opticonn.py apply --config results/sweep_run_<id>/optimize/optimization_results/top3_candidates.json --data /data"
echo ""
echo -e "${GREEN}🎉 OptiConn is installed.${NC}"

# Show current environment status
echo ""
echo -e "${BLUE}📊 Current Environment Status:${NC}"
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✅ Virtual environment: $VIRTUAL_ENV"
    echo "✅ Python location: $(which python)"
    echo "✅ Python version: $(python --version)"
else
    echo "❌ Virtual environment: Not activated"
fi

if [[ -n "$DSI_STUDIO_CMD" ]]; then
    echo "✅ DSI Studio: $DSI_STUDIO_CMD"
fi
if command -v tckgen &> /dev/null; then
    echo "✅ MRtrix3: $(command -v tckgen)"
elif [[ "$VALIDATE_BACKEND" == "mrtrix3" ]]; then
    echo "⚠️  MRtrix3: tckgen not found on PATH"
fi