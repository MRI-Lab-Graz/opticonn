#!/bin/bash

# MRI - Lab Graz
# Karl Koschutnig
# karl.koschutnig@uni-graz.at
#
# install.sh - Braingraph Pipeline Environment Setup
# Description: Creates virtual environment and installs all required packages for the braingraph pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command-line arguments
DSI_STUDIO_PATH=""
MRTRIX_BIN=""
MRTRIX_USE_PATH=false
MRTRIX_INSTALL=false
INSTALL_DOCS=false
DRY_RUN=false

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Backends (pick at least one):"
    echo "  --dsi-path PATH         Path to DSI Studio executable"
    echo "                          Example: /usr/local/bin/dsi_studio"
    echo "                          Or: /Applications/dsi_studio.app/Contents/MacOS/dsi_studio"
    echo "  --mrtrix                Validate MRtrix3 tools from current PATH (tckgen, tck2connectome)"
    echo "  --mrtrix-bin PATH       Path to MRtrix3 bin directory (containing tckgen, tck2connectome, etc.)"
    echo "  --mrtrix-install        Install MRtrix3 (conda-forge) into tools/mrtrix3-conda/ via micromamba"
    echo ""
    echo "Other options:"
    echo "  --docs                  Install MkDocs documentation dependencies (optional)"
    echo "  --dry-run               Validate inputs and print actions without making changes"
    echo "  --help, -h              Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  # DSI Studio backend only"
    echo "  $0 --dsi-path /usr/local/bin/dsi_studio"
    echo ""
    echo "  # MRtrix backend using PATH"
    echo "  $0 --mrtrix"
    echo ""
    echo "  # MRtrix backend using explicit bin directory"
    echo "  $0 --mrtrix-bin /opt/mrtrix3/bin"
    echo ""
    echo "  # Install MRtrix3 locally (no system package manager needed)"
    echo "  $0 --mrtrix-install"
    echo ""
    echo "  # Both backends + docs"
    echo "  $0 --dsi-path /usr/local/bin/dsi_studio --mrtrix-bin /opt/mrtrix3/bin --docs"
}

if [ "$#" -eq 0 ]; then
    print_help
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --dsi-path)
            DSI_STUDIO_PATH="$2"
            shift 2
            ;;
        --mrtrix)
            MRTRIX_USE_PATH=true
            shift
            ;;
        --mrtrix-bin)
            MRTRIX_BIN="$2"
            shift 2
            ;;
        --mrtrix-install)
            MRTRIX_INSTALL=true
            shift
            ;;
        --docs)
            INSTALL_DOCS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

DSI_ENABLED=false
MRTRIX_ENABLED=false
if [ -n "$DSI_STUDIO_PATH" ]; then
    DSI_ENABLED=true
fi
if [ -n "$MRTRIX_BIN" ] || [ "$MRTRIX_USE_PATH" = "true" ] || [ "$MRTRIX_INSTALL" = "true" ]; then
    MRTRIX_ENABLED=true
fi

if [ "$DSI_ENABLED" != "true" ] && [ "$MRTRIX_ENABLED" != "true" ]; then
    echo -e "${RED}❌ Error: select at least one backend (--dsi-path and/or --mrtrix/--mrtrix-bin)${NC}"
    echo ""
    print_help
    exit 1
fi

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
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}DRY-RUN: uv is not installed; would install uv via https://astral.sh/uv/install.sh${NC}"
    else
        echo -e "${RED}❌ uv is not installed. Installing uv...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Ensure current process can find uv without relying on shell rc files
        export PATH="$HOME/.local/bin:$PATH"
        # Best-effort: source common rc files if they exist (do not fail if missing)
        if [ -f "$HOME/.bashrc" ]; then
            # shellcheck disable=SC1090
            source "$HOME/.bashrc" || true
        fi
        if [ -f "$HOME/.zshrc" ]; then
            # shellcheck disable=SC1090
            source "$HOME/.zshrc" || true
        fi
        echo -e "${GREEN}✅ uv installed successfully${NC}"
    fi
else
    echo -e "${GREEN}✅ uv is already installed${NC}"
fi

# Set defaults for uv network behavior (tunable by the user)
# Increase HTTP timeout to reduce network-timeout failures when fetching wheels
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
# Number of attempts to try uv pip install before falling back
export UV_RETRY_COUNT="${UV_RETRY_COUNT:-3}"

# Remove existing virtual environment if it exists
if [ -d "braingraph_pipeline" ]; then
    echo -e "${YELLOW}🗑️  Removing existing virtual environment...${NC}"
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}DRY-RUN: would remove braingraph_pipeline/${NC}"
    else
        rm -rf braingraph_pipeline
    fi
fi

# Create virtual environment
echo -e "${BLUE}📦 Creating virtual environment 'braingraph_pipeline'...${NC}"
if [ "$DRY_RUN" = "true" ]; then
    echo -e "${YELLOW}DRY-RUN: would run: uv venv braingraph_pipeline --python 3.10${NC}"
else
    uv venv braingraph_pipeline --python 3.10
fi

# Activate the virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
if [ "$DRY_RUN" = "true" ]; then
    echo -e "${YELLOW}DRY-RUN: would source braingraph_pipeline/bin/activate${NC}"
else
    source braingraph_pipeline/bin/activate
fi

# Configure environment variables in the virtual environment
echo -e "${BLUE}🔧 Configuring environment variables...${NC}"
if [ "$DRY_RUN" = "true" ]; then
    echo -e "${YELLOW}DRY-RUN: would append environment variables to braingraph_pipeline/bin/activate${NC}"
else
    echo "# Braingraph Pipeline Environment Configuration" >> braingraph_pipeline/bin/activate
    echo "export PYTHONPATH=\"$PWD:\$PYTHONPATH\"" >> braingraph_pipeline/bin/activate
    echo "export TMPDIR=/data/local/tmp_big" >> braingraph_pipeline/bin/activate
    echo "export TEMP=/data/local/tmp_big" >> braingraph_pipeline/bin/activate
    echo "export TMP=/data/local/tmp_big" >> braingraph_pipeline/bin/activate
fi

if [ "$DSI_ENABLED" = "true" ]; then
    # Check DSI Studio installation
    echo -e "${BLUE}🔍 Checking DSI Studio installation...${NC}"
    echo -e "${BLUE}   Path: $DSI_STUDIO_PATH${NC}"

    # Validate the path exists and is executable
    if [ ! -f "$DSI_STUDIO_PATH" ] && [ ! -x "$(command -v "$DSI_STUDIO_PATH" 2>/dev/null)" ]; then
        echo -e "${RED}❌ DSI Studio executable not found at: $DSI_STUDIO_PATH${NC}"
        echo -e "${RED}Installation canceled. Please verify the --dsi-path is correct.${NC}"
        exit 1
    fi

    # Test DSI Studio by running it with --version (use timeout if available)
    echo -e "${BLUE}🔧 Testing DSI Studio functionality...${NC}"
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}DRY-RUN: would run: $DSI_STUDIO_PATH --version${NC}"
    else
        if command -v timeout >/dev/null 2>&1; then
            if ! timeout 10 "$DSI_STUDIO_PATH" --version >/dev/null 2>&1; then
                echo -e "${RED}❌ DSI Studio failed to run or does not support --version${NC}"
                echo -e "${RED}Installation canceled. Please ensure DSI Studio is properly installed and working.${NC}"
                exit 1
            fi
        else
            if ! "$DSI_STUDIO_PATH" --version >/dev/null 2>&1; then
                echo -e "${RED}❌ DSI Studio failed to run or does not support --version${NC}"
                echo -e "${RED}Installation canceled. Please ensure DSI Studio is properly installed and working.${NC}"
                exit 1
            fi
        fi
    fi

    echo -e "${GREEN}✅ DSI Studio validated successfully at: $DSI_STUDIO_PATH${NC}"

    # Store DSI Studio path in the activation script for later use
    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}DRY-RUN: would export DSI_STUDIO_PATH in braingraph_pipeline/bin/activate${NC}"
    else
        echo "# DSI Studio Configuration" >> braingraph_pipeline/bin/activate
        echo "export DSI_STUDIO_PATH=\"$DSI_STUDIO_PATH\"" >> braingraph_pipeline/bin/activate
    fi
fi

if [ "$MRTRIX_ENABLED" = "true" ]; then
    echo -e "${BLUE}🔍 Checking MRtrix3 installation...${NC}"

    if [ "$MRTRIX_INSTALL" = "true" ] && [ -n "$MRTRIX_BIN" ]; then
        echo -e "${RED}❌ Error: --mrtrix-install cannot be combined with --mrtrix-bin${NC}"
        exit 1
    fi

    if [ "$MRTRIX_INSTALL" = "true" ]; then
        MRTRIX_CONDA_PREFIX="$PWD/tools/mrtrix3-conda"
        MICROMAMBA_BIN="$PWD/tools/bin/micromamba"

        if [ "$DRY_RUN" = "true" ]; then
            echo -e "${YELLOW}DRY-RUN: would install micromamba to tools/bin/micromamba${NC}"
            echo -e "${YELLOW}DRY-RUN: would run: micromamba create -y -p \"$MRTRIX_CONDA_PREFIX\" -c conda-forge mrtrix3${NC}"
        else
            mkdir -p "$PWD/tools/bin"
            if [ ! -x "$MICROMAMBA_BIN" ]; then
                echo -e "${BLUE}📥 Downloading micromamba (local) ...${NC}"
                tmpdir=$(mktemp -d)
                curl -LsSf "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "$tmpdir/micromamba.tar.bz2"
                tar -xjf "$tmpdir/micromamba.tar.bz2" -C "$tmpdir"
                if [ ! -f "$tmpdir/bin/micromamba" ]; then
                    echo -e "${RED}❌ micromamba download/extract failed (bin/micromamba not found)${NC}"
                    exit 1
                fi
                mv "$tmpdir/bin/micromamba" "$MICROMAMBA_BIN"
                chmod +x "$MICROMAMBA_BIN"
                rm -rf "$tmpdir"
            fi

            echo -e "${BLUE}📦 Installing MRtrix3 into $MRTRIX_CONDA_PREFIX (conda-forge)...${NC}"
            "$MICROMAMBA_BIN" create -y -p "$MRTRIX_CONDA_PREFIX" -c conda-forge mrtrix3
        fi

        MRTRIX_BIN="$MRTRIX_CONDA_PREFIX/bin"
    fi

    tckgen_cmd="tckgen"
    tck2connectome_cmd="tck2connectome"
    mrinfo_cmd="mrinfo"

    if [ -n "$MRTRIX_BIN" ]; then
        if [ ! -d "$MRTRIX_BIN" ]; then
            echo -e "${RED}❌ MRtrix bin directory not found: $MRTRIX_BIN${NC}"
            exit 1
        fi
        tckgen_cmd="$MRTRIX_BIN/tckgen"
        tck2connectome_cmd="$MRTRIX_BIN/tck2connectome"
        mrinfo_cmd="$MRTRIX_BIN/mrinfo"
    fi

    if [ ! -x "$tckgen_cmd" ] && ! command -v "$tckgen_cmd" >/dev/null 2>&1; then
        echo -e "${RED}❌ MRtrix tool not found/executable: tckgen${NC}"
        echo -e "${YELLOW}Tip: pass --mrtrix-bin /path/to/mrtrix3/bin or ensure MRtrix is on PATH.${NC}"
        exit 1
    fi
    if [ ! -x "$tck2connectome_cmd" ] && ! command -v "$tck2connectome_cmd" >/dev/null 2>&1; then
        echo -e "${RED}❌ MRtrix tool not found/executable: tck2connectome${NC}"
        echo -e "${YELLOW}Tip: pass --mrtrix-bin /path/to/mrtrix3/bin or ensure MRtrix is on PATH.${NC}"
        exit 1
    fi

    if [ "$DRY_RUN" = "true" ]; then
        echo -e "${YELLOW}DRY-RUN: would run: $mrinfo_cmd -version${NC}"
    else
        if [ -x "$mrinfo_cmd" ]; then
            "$mrinfo_cmd" -version >/dev/null 2>&1 || true
        else
            mrinfo -version >/dev/null 2>&1 || true
        fi
    fi

    echo -e "${GREEN}✅ MRtrix3 tools available (tckgen, tck2connectome).${NC}"

    if [ "$DRY_RUN" = "true" ]; then
        if [ -n "$MRTRIX_BIN" ]; then
            echo -e "${YELLOW}DRY-RUN: would prepend MRTRIX_BIN to PATH in braingraph_pipeline/bin/activate${NC}"
        else
            echo -e "${YELLOW}DRY-RUN: MRtrix on PATH; no activate PATH changes needed${NC}"
        fi
    else
        echo "# MRtrix3 Configuration" >> braingraph_pipeline/bin/activate
        if [ -n "$MRTRIX_BIN" ]; then
            echo "export MRTRIX_BIN=\"$MRTRIX_BIN\"" >> braingraph_pipeline/bin/activate
            echo "export PATH=\"$MRTRIX_BIN:\$PATH\"" >> braingraph_pipeline/bin/activate
        fi
    fi
fi

if [ "$DRY_RUN" = "true" ]; then
    echo -e "${GREEN}✅ DRY-RUN completed: inputs validated; no changes were made.${NC}"
    exit 0
fi

echo -e "${BLUE}📦 Installing OptiConn and dependencies (editable, with dev extras)...${NC}"

# Try installing with uv (which uses local cache and retries) with a retry loop.
# If all attempts fail, fall back to pip inside the activated venv.
uv_success=false
attempt=1
while [ "$attempt" -le "$UV_RETRY_COUNT" ]; do
    echo -e "${BLUE}🔁 Attempt $attempt of $UV_RETRY_COUNT using uv to install packages (timeout=${UV_HTTP_TIMEOUT}s)...${NC}"
    if uv pip install -e ".[dev]"; then
        echo -e "${GREEN}✅ Package installation completed successfully using uv!${NC}"
        uv_success=true
        break
    else
        echo -e "${YELLOW}⚠️ uv install attempt $attempt failed. Retrying in 5s...${NC}"
        attempt=$((attempt+1))
        sleep 5
    fi
done

if [ "$uv_success" != "true" ]; then
    echo -e "${RED}❌ All uv attempts failed. Falling back to pip inside the virtualenv...${NC}"
    echo -e "${BLUE}🔧 Ensuring pip, setuptools and wheel are up-to-date in the venv...${NC}"
    python -m pip install --upgrade pip setuptools wheel || true

    echo -e "${BLUE}📦 Running fallback: python -m pip install -e \".[dev]\"${NC}"
    if python -m pip install -e ".[dev]"; then
        echo -e "${GREEN}✅ Package installation completed successfully using pip fallback.${NC}"
    else
        echo -e "${RED}❌ pip fallback also failed. Possible causes: network issues, corrupted cache, or transient PyPI failures.${NC}"
        echo -e "${YELLOW}Tip: Try increasing UV_HTTP_TIMEOUT or UV_RETRY_COUNT and re-run the script. For example:${NC}"
        echo -e "  export UV_HTTP_TIMEOUT=300"
        echo -e "  export UV_RETRY_COUNT=5"
        echo -e "Then re-run: ./install.sh"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✅ Package installation completed successfully!${NC}"

if [ "$INSTALL_DOCS" = "true" ]; then
    echo -e "${BLUE}📚 Installing documentation dependencies (docs/requirements.txt)...${NC}"
    if uv pip install -r docs/requirements.txt; then
        echo -e "${GREEN}✅ Documentation dependencies installed successfully using uv!${NC}"
    else
        echo -e "${YELLOW}⚠️ uv failed to install docs dependencies. Falling back to pip...${NC}"
        python -m pip install -r docs/requirements.txt
        echo -e "${GREEN}✅ Documentation dependencies installed successfully using pip.${NC}"
    fi
fi
echo ""
echo -e "${BLUE}🎯 Environment Summary:${NC}"
echo "• Virtual environment: braingraph_pipeline/"
echo "• Python version: 3.10"
echo "• OptiConn installed in editable mode with dev extras"
echo "• Optimization and sensitivity analysis features available"
if [ "$DSI_ENABLED" = "true" ]; then
echo "• DSI Studio configured (DSI_STUDIO_PATH)"
fi
if [ "$MRTRIX_ENABLED" = "true" ]; then
echo "• MRtrix3 validated (tckgen, tck2connectome)"
if [ -n "$MRTRIX_BIN" ]; then
echo "• MRtrix bin path exported (MRTRIX_BIN)"
fi
fi
if [ "${INSTALL_DOCS}" = "true" ]; then
echo "• MkDocs documentation dependencies installed"
fi
echo ""
echo -e "${YELLOW}📋 To activate the environment:${NC}"
echo "  source braingraph_pipeline/bin/activate"
echo ""
echo -e "${YELLOW}📋 To deactivate the environment:${NC}"
echo "  deactivate"
echo ""
echo -e "${GREEN}🚀 Environment ready for braingraph pipeline!${NC}"
