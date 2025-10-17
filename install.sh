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
    rm -rf braingraph_pipeline
fi

# Create virtual environment
echo -e "${BLUE}📦 Creating virtual environment 'braingraph_pipeline'...${NC}"
uv venv braingraph_pipeline --python 3.10

# Activate the virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source braingraph_pipeline/bin/activate

echo -e "${BLUE}📦 Installing OptiConn and dependencies (editable, with dev and bayesian extras)...${NC}"

# Try installing with uv (which uses local cache and retries) with a retry loop.
# If all attempts fail, fall back to pip inside the activated venv.
uv_success=false
attempt=1
while [ "$attempt" -le "$UV_RETRY_COUNT" ]; do
    echo -e "${BLUE}🔁 Attempt $attempt of $UV_RETRY_COUNT using uv to install packages (timeout=${UV_HTTP_TIMEOUT}s)...${NC}"
    if uv pip install -e ".[dev,bayesian]"; then
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

    echo -e "${BLUE}📦 Running fallback: python -m pip install -e \".[dev,bayesian]\"${NC}"
    if python -m pip install -e ".[dev,bayesian]"; then
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

echo -e "${BLUE}📦 Installing Dash and Plotly for dash_app...${NC}"
uv pip install dash plotly

echo ""
echo -e "${GREEN}✅ Package installation completed successfully!${NC}"
echo ""
echo -e "${BLUE}🎯 Environment Summary:${NC}"
echo "• Virtual environment: braingraph_pipeline/"
echo "• Python version: 3.10"
echo "• OptiConn installed in editable mode with dev and bayesian extras"
echo "• Bayesian optimization and sensitivity analysis features available"
echo ""
echo -e "${YELLOW}📋 To activate the environment:${NC}"
echo "  source braingraph_pipeline/bin/activate"
echo ""
echo -e "${YELLOW}📋 To deactivate the environment:${NC}"
echo "  deactivate"
echo ""
echo -e "${GREEN}🚀 Environment ready for braingraph pipeline!${NC}"
