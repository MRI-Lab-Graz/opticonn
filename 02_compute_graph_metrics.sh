#!/bin/bash

# Graph Metrics Collection Script
# ===============================
# 
# Collects and consolidates pre-computed graph metrics from DSI Studio output
# Uses pipeline metadata from Step 01 for seamless integration
#
# Usage: ./02_compute_graph_metrics.sh [--input pipeline_metadata.json] [--output output_dir]
#
# Author: Braingraph Pipeline Team

set -euo pipefail

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/braingraph_pipeline"

# Default paths
DEFAULT_OUTPUT_BASE="$SCRIPT_DIR/graph_metrics"

# Parse arguments
PIPELINE_METADATA=""
OUTPUT_BASE="$DEFAULT_OUTPUT_BASE"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            PIPELINE_METADATA="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--input pipeline_metadata.json] [--output output_dir]"
            echo ""
            echo "Options:"
            echo "  -i, --input     Pipeline metadata JSON file from Step 01"
            echo "  -o, --output    Output directory for collected metrics"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "If no input is specified, will auto-detect the most recent pipeline metadata."
            exit 0
            ;;
        *)
            # Backward compatibility: treat first argument as pipeline metadata
            if [ -z "$PIPELINE_METADATA" ]; then
                PIPELINE_METADATA="$1"
            fi
            shift
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "📊 Graph Metrics Collection Pipeline"
echo "===================================="
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

# Auto-detect pipeline metadata if not provided
if [ -z "$PIPELINE_METADATA" ]; then
    echo -e "${BLUE}🔍 Auto-detecting pipeline metadata...${NC}"
    
    # Look for the most recent pipeline metadata file
    RECENT_METADATA=""
    
    # Search in common locations
    for search_pattern in "*/pipeline_metadata_*.json" "*/organized_matrices/pipeline_metadata_*.json" "pipeline_metadata_*.json"; do
        LATEST=$(find "$SCRIPT_DIR" -name "$(basename "$search_pattern")" -type f 2>/dev/null | sort | tail -1)
        if [ -n "$LATEST" ]; then
            RECENT_METADATA="$LATEST"
            break
        fi
    done
    
    if [ -z "$RECENT_METADATA" ]; then
        echo -e "${RED}❌ No pipeline metadata files found${NC}"
        echo "Please run 01_extract_connectome.sh first, or specify metadata file:"
        echo "Usage: $0 --input <pipeline_metadata.json>"
        exit 1
    fi
    
    PIPELINE_METADATA="$RECENT_METADATA"
fi

# Validate pipeline metadata file
if [ ! -f "$PIPELINE_METADATA" ]; then
    echo -e "${RED}❌ Pipeline metadata file not found: $PIPELINE_METADATA${NC}"
    exit 1
fi

echo "� Input metadata: $PIPELINE_METADATA"
echo "📁 Output directory: $OUTPUT_BASE"
echo ""

# Validate JSON format and extract information
echo -e "${BLUE}🔍 Validating pipeline metadata...${NC}"
METADATA_INFO=$(python -c "
import json
import sys
import os

try:
    with open('$PIPELINE_METADATA', 'r') as f:
        metadata = json.load(f)
    
    if 'pipeline_metadata' not in metadata:
        print('❌ Invalid metadata format: missing pipeline_metadata section')
        sys.exit(1)
    
    pm = metadata['pipeline_metadata']
    
    # Extract key information
    step = pm.get('step', 'unknown')
    timestamp = pm.get('timestamp', 'unknown')
    organized_dir = pm.get('results', {}).get('organized_matrices_directory', '')
    atlas_count = pm.get('results', {}).get('atlases_processed', 0)
    
    print(f'✅ Valid pipeline metadata')
    print(f'Step: {step}')
    print(f'Timestamp: {timestamp}')
    print(f'Organized directory: {organized_dir}')
    print(f'Atlases processed: {atlas_count}')
    
    # Check if organized directory exists
    if organized_dir and os.path.exists(organized_dir):
        print(f'✅ Organized directory exists')
        
        # Count network measures files in the organized structure
        # Structure: organized_dir/{atlas}/{metric}/*.network_measures.txt
        measures_count = 0
        
        for root, dirs, files in os.walk(organized_dir):
            measures_count += len([f for f in files if f.endswith('network_measures.txt')])
        
        print(f'Found {measures_count} network measures files')
        
        if measures_count == 0:
            print('❌ No network measures files found')
            sys.exit(1)
        
        # Output information for shell script
        print(f'ORGANIZED_DIR={organized_dir}')
        print(f'MEASURES_COUNT={measures_count}')
        print(f'TIMESTAMP={timestamp}')
        
    else:
        print(f'❌ Organized directory not found: {organized_dir}')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Error processing metadata: {e}')
    sys.exit(1)
")

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Pipeline metadata validation failed${NC}"
    exit 1
fi

# Extract information from validation output
ORGANIZED_DIR=$(echo "$METADATA_INFO" | grep "ORGANIZED_DIR=" | cut -d'=' -f2-)
MEASURES_COUNT=$(echo "$METADATA_INFO" | grep "MEASURES_COUNT=" | cut -d'=' -f2-)
METADATA_TIMESTAMP=$(echo "$METADATA_INFO" | grep "TIMESTAMP=" | cut -d'=' -f2-)

echo -e "${GREEN}📁 Using organized data from: $ORGANIZED_DIR${NC}"
echo -e "${BLUE}📈 Found $MEASURES_COUNT network measures files${NC}"

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
METRICS_OUTPUT="$OUTPUT_BASE/collected_metrics_$TIMESTAMP"
mkdir -p "$METRICS_OUTPUT"

echo -e "${YELLOW}📊 Collecting pre-computed graph metrics...${NC}"
echo "Output directory: $METRICS_OUTPUT"
echo ""

# Create metrics collection script with absolute paths
cat > "$METRICS_OUTPUT/collect_metrics.py" << EOF
#!/usr/bin/env python3
"""
Graph metrics collection from DSI Studio pre-computed network measures.
Integrated with pipeline metadata from Step 01.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metrics_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_network_measures_file(file_path: Path) -> Dict:
    """Parse a DSI Studio network_measures.txt file."""
    metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        key, value = parts
                        try:
                            # Convert to float if possible
                            metrics[key] = float(value)
                        except ValueError:
                            metrics[key] = value
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
    
    return metrics

def extract_metadata_from_path(file_path: Path) -> Dict:
    """Extract subject, atlas, and connectivity metric from file path."""
    metadata = {
        'subject': 'unknown',
        'atlas': 'unknown', 
        'connectivity_metric': 'unknown',
        'file_path': str(file_path)
    }
    
    # Parse organized structure: organized_matrices/{atlas}/{metric}/filename.txt
    # Filename pattern: sub-{ID}_timestamp_{atlas}_{metric}_network_measures.txt
    filename = file_path.name
    path_parts = file_path.parts
    
    # Extract subject ID from filename
    sub_match = re.search(r'(sub-[^_]+)', filename)
    if sub_match:
        metadata['subject'] = sub_match.group(1)
    
    # Extract atlas and metric from path structure
    # Find organized_matrices in path, then atlas and metric follow
    for i, part in enumerate(path_parts):
        if part == 'organized_matrices' and i + 2 < len(path_parts):
            metadata['atlas'] = path_parts[i + 1]
            metadata['connectivity_metric'] = path_parts[i + 2]
            break
        elif 'organized_matrices' in part and i + 1 < len(path_parts):
            # Handle case where organized_matrices is part of a longer path
            metadata['atlas'] = path_parts[i + 1]
            if i + 2 < len(path_parts):
                metadata['connectivity_metric'] = path_parts[i + 2]
            break
    
    # Alternative: extract from filename if path parsing failed
    if metadata['atlas'] == 'unknown' or metadata['connectivity_metric'] == 'unknown':
        # Filename pattern: sub-{ID}_timestamp_{atlas}_{metric}_network_measures.txt
        parts = filename.replace('.txt', '').split('_')
        if len(parts) >= 4:
            # Find network_measures position and work backwards
            if 'network' in parts and 'measures' in parts:
                measures_idx = -1
                for i, part in enumerate(parts):
                    if part == 'network' and i + 1 < len(parts) and parts[i + 1] == 'measures':
                        measures_idx = i
                        break
                
                if measures_idx >= 2:
                    metadata['connectivity_metric'] = parts[measures_idx - 1]
                    metadata['atlas'] = '_'.join(parts[2:measures_idx - 1])
    
    return metadata

def standardize_column_names(metrics: Dict) -> Dict:
    """Standardize metric column names for consistency."""
    name_mapping = {
        # Density/Sparsity
        'density': 'density',
        
        # Clustering
        'clustering_coeff_average(binary)': 'clustering_coefficient_binary',
        'clustering_coeff_average(weighted)': 'clustering_coefficient_weighted',
        'transitivity(binary)': 'transitivity_binary', 
        'transitivity(weighted)': 'transitivity_weighted',
        
        # Path length
        'network_characteristic_path_length(binary)': 'characteristic_path_length_binary',
        'network_characteristic_path_length(weighted)': 'characteristic_path_length_weighted',
        
        # Small-worldness
        'small-worldness(binary)': 'small_worldness_binary',
        'small-worldness(weighted)': 'small_worldness_weighted',
        
        # Efficiency
        'global_efficiency(binary)': 'global_efficiency_binary',
        'global_efficiency(weighted)': 'global_efficiency_weighted',
        
        # Graph properties
        'diameter_of_graph(binary)': 'diameter_binary',
        'diameter_of_graph(weighted)': 'diameter_weighted',
        'radius_of_graph(binary)': 'radius_binary',
        'radius_of_graph(weighted)': 'radius_weighted',
        
        # Assortativity
        'assortativity_coefficient(binary)': 'assortativity_binary',
        'assortativity_coefficient(weighted)': 'assortativity_weighted',
    }
    
    standardized = {}
    for key, value in metrics.items():
        new_key = name_mapping.get(key, key)
        standardized[new_key] = value
    
    # Add computed metrics
    if 'density' in standardized:
        standardized['sparsity'] = 1.0 - standardized['density']
    
    # Use weighted metrics as defaults (these vary between connectivity metrics)
    if 'clustering_coefficient_weighted' in standardized:
        standardized['clustering_coefficient'] = standardized['clustering_coefficient_weighted']
    if 'characteristic_path_length_weighted' in standardized:
        standardized['characteristic_path_length'] = standardized['characteristic_path_length_weighted'] 
    if 'small_worldness_weighted' in standardized:
        standardized['small_worldness'] = standardized['small_worldness_weighted']
    if 'global_efficiency_weighted' in standardized:
        standardized['global_efficiency'] = standardized['global_efficiency_weighted']
    if 'assortativity_weighted' in standardized:
        standardized['assortativity'] = standardized['assortativity_weighted']
    
    return standardized

def main():
    if len(sys.argv) != 4:
        print("Usage: python collect_metrics.py <pipeline_metadata.json> <organized_dir> <output_dir>")
        sys.exit(1)
    
    metadata_file = Path(sys.argv[1])
    organized_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    
    logger.info(f"Pipeline metadata: {metadata_file}")
    logger.info(f"Organized directory: {organized_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load pipeline metadata
    try:
        with open(metadata_file, 'r') as f:
            pipeline_metadata = json.load(f)
        logger.info("✅ Pipeline metadata loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline metadata: {e}")
        sys.exit(1)
    
    # Find all network measures files using the actual organized structure
    # Structure: organized_dir/{atlas}/{metric}/*.network_measures.txt
    measures_files = list(organized_dir.glob("**/*network_measures.txt"))
    logger.info(f"Found {len(measures_files)} network measures files")
    
    if not measures_files:
        logger.error("❌ No network measures files found")
        sys.exit(1)
    
    # Collect all metrics
    all_metrics = []
    
    for measures_file in measures_files:
        logger.info(f"Processing: {measures_file}")
        
        try:
            # Parse metrics from file
            metrics = parse_network_measures_file(measures_file)
            
            # Extract metadata
            metadata = extract_metadata_from_path(measures_file)
            
            # Standardize column names
            standardized_metrics = standardize_column_names(metrics)
            
            # Combine metadata and metrics
            combined = {**metadata, **standardized_metrics}
            all_metrics.append(combined)
            
        except Exception as e:
            logger.error(f"Error processing {measures_file}: {e}")
            continue
    
    if not all_metrics:
        logger.error("No metrics collected!")
        sys.exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Clean up unknown values
    df = df[df['atlas'] != 'unknown']
    df = df[df['connectivity_metric'] != 'unknown']
    
    logger.info(f"Collected metrics for {len(df)} combinations")
    logger.info(f"Unique subjects: {df['subject'].nunique()}")
    logger.info(f"Unique atlases: {df['atlas'].nunique()}")
    logger.info(f"Unique connectivity metrics: {df['connectivity_metric'].nunique()}")
    
    # Save comprehensive metrics
    output_file = output_dir / "graph_metrics_comprehensive.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved comprehensive metrics to: {output_file}")
    
    # Save global metrics only (excluding detailed/nodal metrics)
    global_cols = [col for col in df.columns if not any(x in col.lower() for x in ['nodal', 'node_', 'rich_club'])]
    global_df = df[global_cols]
    global_output = output_dir / "graph_metrics_global.csv"
    global_df.to_csv(global_output, index=False)
    logger.info(f"Saved global metrics to: {global_output}")
    
    # Create enhanced summary statistics with pipeline integration
    summary = {
        'collection_timestamp': pd.Timestamp.now().isoformat(),
        'input_pipeline_metadata': str(metadata_file),
        'source_pipeline_step': pipeline_metadata.get('pipeline_metadata', {}).get('step', 'unknown'),
        'source_timestamp': pipeline_metadata.get('pipeline_metadata', {}).get('timestamp', 'unknown'),
        'total_files_processed': len(measures_files),
        'total_combinations': len(df),
        'unique_subjects': int(df['subject'].nunique()),
        'unique_atlases': int(df['atlas'].nunique()),
        'unique_connectivity_metrics': int(df['connectivity_metric'].nunique()),
        'subjects': sorted(df['subject'].unique().tolist()),
        'atlases': sorted(df['atlas'].unique().tolist()),
        'connectivity_metrics': sorted(df['connectivity_metric'].unique().tolist()),
        'available_metrics': [col for col in df.columns if col not in ['subject', 'atlas', 'connectivity_metric', 'file_path']],
        'ready_for_optimization': {
            'input_file': str(global_output),
            'next_step': '03_optimize_metrics.sh',
            'command': f'./03_optimize_metrics.sh {global_output}'
        }
    }
    
    summary_file = output_dir / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Collection complete! Results saved to: {output_dir}")
    logger.info(f"Ready for metric optimization with: {global_output}")

if __name__ == "__main__":
    main()
EOF

# Run the metrics collection
echo -e "${YELLOW}🔄 Starting metrics collection...${NC}"


# Absoluten Pfad für organized_dir vor dem cd berechnen
ABS_ORGANIZED_DIR="$(realpath "$ORGANIZED_DIR")"
cd "$METRICS_OUTPUT"
python collect_metrics.py "$PIPELINE_METADATA" "$ABS_ORGANIZED_DIR" "$METRICS_OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Graph metrics collection completed successfully!${NC}"
    echo ""
    echo "📊 Results summary:"
    echo "=================="
    
    # Show summary if available
    if [ -f "$METRICS_OUTPUT/collection_summary.json" ]; then
        python -c "
import json
with open('$METRICS_OUTPUT/collection_summary.json', 'r') as f:
    summary = json.load(f)
print(f\"Files processed: {summary['total_files_processed']}\")
print(f\"Total combinations: {summary['total_combinations']}\")
print(f\"Unique subjects: {summary['unique_subjects']}\")
print(f\"Unique atlases: {summary['unique_atlases']}\")
print(f\"Unique connectivity metrics: {summary['unique_connectivity_metrics']}\")
print(f\"Atlases: {', '.join(summary['atlases'])}\")
print(f\"Connectivity metrics: {', '.join(summary['connectivity_metrics'])}\")
print(f\"Available graph metrics: {len(summary['available_metrics'])} metrics\")
"
    fi
    
    echo ""
    echo "📁 Output files:"
    echo "  - Comprehensive metrics: $METRICS_OUTPUT/graph_metrics_comprehensive.csv"
    echo "  - Global metrics only: $METRICS_OUTPUT/graph_metrics_global.csv"
    echo "  - Collection summary: $METRICS_OUTPUT/collection_summary.json"
    echo "  - Logs: $METRICS_OUTPUT/metrics_collection.log"
    echo ""
    echo "🚀 Ready for metric optimization! Run:"
    # --- Automatische Optimierungs-Vorbereitung (ehemals Schritt 03) ---
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OPTIMIZATION_DIR="$SCRIPT_DIR/optimization_results/optimization_$TIMESTAMP"
    mkdir -p "$OPTIMIZATION_DIR"

    # Finde die neueste graph_metrics_global.csv
    METRICS_FILE=$(ls -1dt $METRICS_OUTPUT/graph_metrics_global.csv 2>/dev/null | head -1)
    if [ ! -f "$METRICS_FILE" ]; then
        echo -e "${RED}❌ Keine graph_metrics_global.csv gefunden!${NC}"
        exit 1
    fi
    cp "$METRICS_FILE" "$OPTIMIZATION_DIR/optimized_metrics.csv"

    # quality_score berechnen
    echo -e "${BLUE}🧮 Berechne quality_score für Optimierung...${NC}"
    source "$VENV_DIR/bin/activate"
    python "$SCRIPT_DIR/metric_optimizer.py" "$OPTIMIZATION_DIR/optimized_metrics.csv" "$OPTIMIZATION_DIR" --plots

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Optimierungs-Vorbereitung abgeschlossen!${NC}"
        echo "  - Optimized metrics: $OPTIMIZATION_DIR/optimized_metrics.csv"
        echo "  - Report: $OPTIMIZATION_DIR/optimization_report.txt"
        echo "  - Plots: $OPTIMIZATION_DIR/*.png"
        echo "  - Weiter mit: ./03_balanced_optimizer.sh"
    else
        echo -e "${RED}❌ Optimierungs-Vorbereitung fehlgeschlagen!${NC}"
        exit 1
    fi
    
else
    echo -e "${RED}❌ Graph metrics collection failed${NC}"
    exit 1
fi
