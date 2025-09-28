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

echo -e "${BLUE}📦 Installing OptiConn and dependencies (editable, with dev extras)...${NC}"
uv pip install -e ".[dev]"

echo ""
echo -e "${GREEN}✅ Package installation completed successfully!${NC}"
echo ""
echo -e "${BLUE}🎯 Environment Summary:${NC}"
echo "• Virtual environment: braingraph_pipeline/"
echo "• Python version: 3.10"
echo "• OptiConn installed in editable mode with dev extras"
echo ""
echo -e "${YELLOW}📋 To activate the environment:${NC}"
echo "  source braingraph_pipeline/bin/activate"
echo ""
echo -e "${YELLOW}📋 To deactivate the environment:${NC}"
echo "  deactivate"
echo ""
echo -e "${GREEN}🚀 Environment ready for braingraph pipeline!${NC}"
