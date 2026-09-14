#!/bin/bash

# OptiConn Environment Activation Script
# Source this script to activate the OptiConn environment

# Find the braingraph-pipeline directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/../braingraph-pipeline"

if [[ ! -d "$PIPELINE_DIR" ]]; then
    echo "❌ braingraph-pipeline directory not found at: $PIPELINE_DIR"
    echo "💡 Expected structure:"
    echo "   parent/"
    echo "   ├── braingraph-pipeline/"
    echo "   └── opticonn/"
    return 1 2>/dev/null || exit 1
fi

# Check for virtual environment
VENV_PATH="$PIPELINE_DIR/braingraph_pipeline"
if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo "💡 Run ./install.sh first to set up the environment"
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
echo "🔧 Activating OptiConn environment..."
source "$VENV_PATH/bin/activate"

if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✅ OptiConn environment activated"
    echo "📍 Python: $(which python)"
    echo "📍 Version: $(python --version)"
    
    # Check for DSI Studio
    if [[ -z "$DSI_STUDIO_CMD" ]]; then
        echo ""
        echo "⚠️  DSI Studio not configured"
        echo "💡 Set with: export DSI_STUDIO_CMD=/path/to/dsi_studio"
        echo ""
        
        # Try to find DSI Studio automatically
        DSI_LOCATIONS=(
            "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio"
            "/usr/local/bin/dsi_studio"
            "/opt/dsi_studio/dsi_studio"
        )
        
        for location in "${DSI_LOCATIONS[@]}"; do
            if [[ -f "$location" ]]; then
                echo "🔍 Found DSI Studio at: $location"
                echo "💡 Run: export DSI_STUDIO_CMD=\"$location\""
                break
            fi
        done
    else
        echo "✅ DSI Studio: $DSI_STUDIO_CMD"
    fi
    
    echo ""
    echo "🚀 Ready to use OptiConn!"
    echo "   Test: python opticonn.py validate"
    echo "   Quick start: python opticonn.py sweep --data /path/to/subjects --quick"
    
else
    echo "❌ Failed to activate virtual environment"
    return 1 2>/dev/null || exit 1
fi