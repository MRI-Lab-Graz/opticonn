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

echo "📊 Found $(ls data/*.fz data/*.fib.gz 2>/dev/null | wc -l) subject files"
echo ""

# Activate environment (use the local venv created by install.sh)
echo "🔧 Activating environment..."
if [[ -f "../../.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ../../.venv/bin/activate
else
    echo "⚠️  Local virtualenv not found at ../../.venv — make sure to run ../install.sh"
fi

# Run OptiConn in quick mode (use local opticonn entrypoint)
echo "🚀 Running OptiConn quick test..."
python3 ../../opticonn.py sweep \
    --data data \
    --output results \
    --quick \
    --subjects 2

echo ""
echo "✅ Quick start example completed!"
echo "📁 Check results/ directory for outputs"
