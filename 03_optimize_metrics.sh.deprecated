#!/bin/bash
#
# WORKFLOW MIGRATION NOTICE - DEPRECATED SCRIPT
# ==============================================
#
# This file has been DEPRECATED and replaced with 04_balanced_optimizer.sh
#
# REASON FOR DEPRECATION:
# The original optimization favored FA metrics with extreme effect sizes 
# (Cohen's d > 15), which dominate atlas selection but are not optimal 
# for biological group comparisons in soccer neuroscience studies.
#
# NEW WORKFLOW:
# Use 04_balanced_optimizer.sh instead, which:
# - Penalizes FA dominance (extreme effect sizes)
# - Favors COUNT/NCOUNT2 (moderate effect sizes 0.8-3.0)
# - Maintains scientific rigor for group studies
# - Better suited for soccer vs control comparisons
#
# MIGRATION:
# Simply run: ./04_balanced_optimizer.sh
#
# Date deprecated: 2025-08-07
#

echo "❌ DEPRECATED SCRIPT"
echo "==================="
echo ""
echo "This script (03_optimize_metrics.sh) has been deprecated."
echo ""
echo "🔄 PLEASE USE: ./04_balanced_optimizer.sh"
echo ""
echo "Reason: The original optimization favored FA metrics with extreme"
echo "effect sizes that dominate but are not optimal for group studies."
echo ""
echo "The new balanced optimizer addresses this issue while maintaining"
echo "scientific rigor for your soccer neuroscience study."
echo ""
echo "Run this command instead:"
echo "  ./04_balanced_optimizer.sh"
echo ""
exit 1
DEFAULT_OUTPUT_BASE="$SCRIPT_DIR/optimization_results"

# Parse arguments
METRICS_FILE="${1:-$DEFAULT_METRICS_FILE}"
OUTPUT_BASE="${2:-$DEFAULT_OUTPUT_BASE}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🎯 Connectivity Metric Optimization"
echo "==================================="
echo ""

# Validate environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV_DIR${NC}"
    echo "Please run ./install.sh first."
    exit 1
fi

# Activate environment
echo -e "${BLUE}🔧 Activating Python environment...${NC}"
source "$VENV_DIR/bin/activate"

# Auto-detect metrics file if not provided
if [ -z "$METRICS_FILE" ]; then
    echo -e "${BLUE}🔍 Auto-detecting graph metrics file...${NC}"
    
    # Look for collection summary first (contains path to metrics file)
    RECENT_SUMMARY=$(find "$SCRIPT_DIR" -name "collection_summary.json" -type f | sort | tail -1)
    
    if [ -n "$RECENT_SUMMARY" ]; then
        echo -e "${BLUE}📄 Found collection summary: $RECENT_SUMMARY${NC}"
        
        # Extract metrics file path from summary
        METRICS_FROM_SUMMARY=$(python -c "
import json
try:
    with open('$RECENT_SUMMARY', 'r') as f:
        summary = json.load(f)
    metrics_file = summary.get('ready_for_optimization', {}).get('input_file', '')
    if metrics_file:
        print(metrics_file)
    else:
        print('')
except:
    print('')
")
        
        if [ -n "$METRICS_FROM_SUMMARY" ] && [ -f "$METRICS_FROM_SUMMARY" ]; then
            METRICS_FILE="$METRICS_FROM_SUMMARY"
            echo -e "${GREEN}✅ Using metrics file from collection summary${NC}"
        fi
    fi
    
    # Fallback: look for the most recent graph metrics file
    if [ -z "$METRICS_FILE" ]; then
        RECENT_METRICS=$(find "$SCRIPT_DIR" -name "*graph_metrics*.csv" -type f | sort | tail -1)
        
        if [ -z "$RECENT_METRICS" ]; then
            echo -e "${RED}❌ No graph metrics files found${NC}"
            echo "Please run 02_compute_graph_metrics.sh first, or specify metrics file:"
            echo "Usage: $0 <metrics_file.csv> [output_dir]"
            exit 1
        fi
        
        METRICS_FILE="$RECENT_METRICS"
    fi
fi

# Validate metrics file
if [ ! -f "$METRICS_FILE" ]; then
    echo -e "${RED}❌ Metrics file not found: $METRICS_FILE${NC}"
    echo "Please run 02_compute_graph_metrics.sh first, or provide a valid metrics file."
    exit 1
fi

echo "📊 Input metrics: $METRICS_FILE"
echo "📁 Output directory: $OUTPUT_BASE"
echo ""

# Validate that metrics file contains required columns
echo -e "${BLUE}🔍 Validating metrics file format...${NC}"
if ! python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$METRICS_FILE')
    print(f'✅ Loaded {len(df)} records from metrics file')
    
    # Check for required columns
    required_cols = ['atlas', 'connectivity_metric']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f'❌ Missing required columns: {missing_cols}')
        print(f'Available columns: {list(df.columns)}')
        sys.exit(1)
    
    print(f'✅ Found required columns: atlas, connectivity_metric')
    print(f'Available atlases: {sorted(df[\"atlas\"].unique())}')
    print(f'Available metrics: {sorted(df[\"connectivity_metric\"].unique())}')
    
except Exception as e:
    print(f'❌ Error reading metrics file: {e}')
    sys.exit(1)
"; then
    echo -e "${RED}❌ Metrics file validation failed${NC}"
    exit 1
fi

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$OUTPUT_BASE/optimization_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}🎯 Starting metric optimization...${NC}"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create optimization configuration if it doesn't exist
OPTIMIZATION_CONFIG="$OUTPUT_DIR/optimization_config.json"
cat > "$OPTIMIZATION_CONFIG" << 'EOF'
{
    "optimization": {
        "metrics_to_evaluate": [
            "sparsity", 
            "small_worldness", 
            "modularity", 
            "global_efficiency", 
            "characteristic_path_length",
            "clustering_coefficient", 
            "assortativity"
        ],
        "weight_factors": {
            "sparsity_score": 0.25,
            "small_worldness": 0.25,
            "modularity": 0.20,
            "global_efficiency": 0.20,
            "reliability": 0.10
        },
        "sparsity_range": [0.05, 0.4],
        "small_world_range": [1.0, 3.0],
        "quality_threshold": 0.65,
        "reliability_threshold": 0.7
    }
}
EOF

echo -e "${YELLOW}📝 Created optimization configuration: $OPTIMIZATION_CONFIG${NC}"

# Run metric optimization
echo -e "${YELLOW}🧮 Running optimization analysis...${NC}"

# Check if metric_optimizer.py exists and is executable
if [ ! -f "$SCRIPT_DIR/metric_optimizer.py" ]; then
    echo -e "${RED}❌ metric_optimizer.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Run the optimization
cd "$OUTPUT_DIR"
python "$SCRIPT_DIR/metric_optimizer.py" \
    "$METRICS_FILE" \
    "$OUTPUT_DIR" \
    --config "$OPTIMIZATION_CONFIG" \
    --plots

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Metric optimization completed successfully!${NC}"
    echo ""
    echo "📊 Results summary:"
    echo "=================="
    
    # Show optimization summary
    if [ -f "$OUTPUT_DIR/optimized_metrics.csv" ]; then
        echo -e "${BLUE}📈 Optimization Statistics:${NC}"
        python -c "
import pandas as pd
import json

# Load optimized results
df = pd.read_csv('$OUTPUT_DIR/optimized_metrics.csv')

print(f'Total combinations analyzed: {len(df)}')

if 'quality_score' in df.columns:
    high_quality = df[df.get('meets_quality_threshold', False)]
    recommended = df[df.get('recommended', False)]
    
    print(f'High-quality combinations: {len(high_quality)}')
    print(f'Recommended combinations: {len(recommended)}')
    print(f'Mean quality score: {df[\"quality_score\"].mean():.3f}')
    print(f'Max quality score: {df[\"quality_score\"].max():.3f}')
    
    print()
    print('🏆 Top 5 combinations:')
    top_5 = df.nlargest(5, 'quality_score')[['atlas', 'connectivity_metric', 'quality_score']]
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f'  {i}. {row[\"atlas\"]} + {row[\"connectivity_metric\"]} (score: {row[\"quality_score\"]:.3f})')
    
    if len(recommended) > 0:
        print()
        print('⭐ Recommended combinations (best per atlas):')
        rec_sorted = recommended.sort_values('quality_score', ascending=False)
        for _, row in rec_sorted.iterrows():
            print(f'  - {row[\"atlas\"]} + {row[\"connectivity_metric\"]} (score: {row[\"quality_score\"]:.3f})')
else:
    print('Quality scores not found in results.')
"
    fi
    
    echo ""
    echo "📁 Output files:"
    echo "  - Optimized metrics: $OUTPUT_DIR/optimized_metrics.csv"
    echo "  - Optimization report: $OUTPUT_DIR/optimization_report.txt"
    echo "  - Configuration: $OPTIMIZATION_CONFIG"
    
    # List plot files if they exist
    PLOT_FILES=$(find "$OUTPUT_DIR" -name "*.png" -type f 2>/dev/null | wc -l)
    if [ "$PLOT_FILES" -gt 0 ]; then
        echo "  - Visualization plots: $OUTPUT_DIR/*.png ($PLOT_FILES files)"
    fi
    
    echo ""
    echo "📋 Next steps:"
    echo "  1. Review optimization report: $OUTPUT_DIR/optimization_report.txt"
    echo "  2. Examine recommended combinations in optimized_metrics.csv"
    echo "  3. Use top-scoring combinations for statistical analysis"
    echo "  4. Consider running cross-validation with selected combinations"
    
else
    echo -e "${RED}❌ Metric optimization failed${NC}"
    echo "Check logs for details."
    exit 1
fi
