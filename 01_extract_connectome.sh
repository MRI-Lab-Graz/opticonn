#!/bin/bash

# ==============================================================================
# 01_extract_connectome.sh
# ==============================================================================
# 
# DSI Studio Connectivity Matrix Extraction
# 
# This script extracts connectivity matrices from .fz files using the existing
# DSI Studio tools. It serves as the first step in the modular pipeline.
#
# Usage:
#   ./01_extract_connectome.sh [options] <input_path> <output_dir>
#
# Author: Braingraph Pipeline Team
# Date: $(date +%Y-%m-%d)
# ==============================================================================

set -e  # Exit on any error

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/braingraph_pipeline"

# Default configuration (now in main directory)
DEFAULT_CONFIG="$SCRIPT_DIR/01_working_config.json"
DEFAULT_ATLASES="ATAG_basal_ganglia,BrainSeg,Brainnectome,Brodmann,Campbell,Cerebellum-SUIT,CerebrA,FreeSurferDKT_Cortical,FreeSurferDKT_Subcortical,FreeSurferDKT_Tissue,FreeSurferSeg,HCP-MMP,HCP842_tractography,HCPex,JulichBrain,Kleist"
DEFAULT_METRICS="count,fa,qa,ncount2"
DEFAULT_TRACKS=4000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Usage function
show_usage() {
    cat << EOF
DSI Studio Connectivity Matrix Extraction - Step 01

Usage: $0 [OPTIONS] INPUT_PATH OUTPUT_DIR

Extract connectivity matrices from .fz files using DSI Studio.

ARGUMENTS:
    INPUT_PATH      Path to .fz file or directory containing .fz files
    OUTPUT_DIR      Directory to save extracted connectivity matrices

OPTIONS:
    -c, --config FILE       Configuration file (default: $DEFAULT_CONFIG)
    -a, --atlases LIST      Comma-separated atlas list (default: $DEFAULT_ATLASES)
    -m, --metrics LIST      Comma-separated metrics list (default: $DEFAULT_METRICS)
    -t, --tracks N          Number of tracks (default: $DEFAULT_TRACKS)
    -p, --pilot             Run pilot test on 1-2 files first
    --pilot-count N         Number of files for pilot test (default: 2)
    --validate              Validate setup before processing
    --organize-only         Only organize existing matrices (skip extraction)
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    # Single file with defaults
    $0 subject001.fz ./results/

    # Directory with custom config
    $0 --config custom_config.json /data/subjects/ ./results/

    # Pilot test first
    $0 --pilot --pilot-count 3 /data/subjects/ ./results/

    # Custom atlases and metrics
    $0 --atlases "AAL3,HCP-MMP" --metrics "count,fa" /data/ ./results/

    # Validate setup first
    $0 --validate --config my_config.json /data/ ./results/

    # Only organize existing extracted matrices
    $0 --organize-only /data/ ./results/

OUTPUT STRUCTURE:
    OUTPUT_DIR/
    ├── connectivity_matrices/          # Raw DSI Studio output
    │   ├── sub-01_ses-1_timestamp/
    │   │   └── tracks_Nk_method/
    │   │       └── by_atlas/
    │   │           └── {atlas}/
    │   └── ...
    ├── organized_matrices/             # Clean, organized structure
    │   ├── by_atlas/
    │   │   ├── {atlas_name}/
    │   │   │   └── {metric_name}/
    │   │   │       └── sub-{ID}_{atlas}_{metric}_{pass}_{type}.txt
    │   │   └── ...
    │   ├── by_subject/
    │   │   ├── {subject_ID}/
    │   │   │   └── all_matrices_for_subject.txt
    │   │   └── ...
    │   ├── matrix_index_YYYYMMDD_HHMMSS.json
    │   └── extraction_summary_YYYYMMDD_HHMMSS.txt
    ├── logs/
    │   └── extraction_YYYYMMDD_HHMMSS.log
    └── summary_YYYYMMDD_HHMMSS.json

EOF
}

# Check if virtual environment exists
check_environment() {
    if [[ ! -d "$VENV_DIR" ]]; then
        error "Virtual environment not found at $VENV_DIR"
        error "Please run ./install.sh first to set up the environment"
        exit 1
    fi
}

# Activate virtual environment
activate_environment() {
    log "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Verify Python environment
    if ! python -c "import sys; print(f'Python {sys.version}')" 2>/dev/null; then
        error "Failed to activate Python environment"
        exit 1
    fi
}

# Validate JSON configuration
validate_config() {
    local config_file="$1"
    
    if [[ -z "$config_file" ]]; then
        return 0  # No config specified, skip validation
    fi
    
    log "Validating configuration file: $config_file"
    
    # Check if json_validator exists
    if [[ ! -f "$SCRIPT_DIR/json_validator.py" ]]; then
        warn "JSON validator not found, skipping configuration validation"
        return 0
    fi
    
    # Check if schema exists
    local schema_file="$SCRIPT_DIR/dsi_studio_config_schema.json"
    if [[ ! -f "$schema_file" ]]; then
        warn "Configuration schema not found, skipping validation"
        return 0
    fi
    
    # Run validation
    if ! python "$SCRIPT_DIR/json_validator.py" "$config_file" --schema "$schema_file"; then
        error "Configuration validation failed"
        error "Please fix the configuration file and try again"
        exit 1
    fi
    
    success "Configuration validation passed"
}

# Validate DSI Studio tools
validate_dsi_tools() {
    log "Validating DSI Studio tools..."
    
    if [[ ! -f "$SCRIPT_DIR/extract_connectivity_matrices.py" ]]; then
        error "DSI Studio extraction script not found: $SCRIPT_DIR/extract_connectivity_matrices.py"
        exit 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/validate_setup.py" ]]; then
        error "DSI Studio validation script not found: $SCRIPT_DIR/validate_setup.py"
        exit 1
    fi
    
    success "DSI Studio tools validated"
}

# Run setup validation
run_validation() {
    local config_file="$1"
    local input_path="$2"
    
    log "Running DSI Studio setup validation..."
    
    local cmd="python $SCRIPT_DIR/validate_setup.py"
    
    if [[ -n "$config_file" ]]; then
        cmd="$cmd --config $config_file"
    fi
    
    if [[ -n "$input_path" ]]; then
        cmd="$cmd --test-input $input_path"
    fi
    
    if ! eval "$cmd"; then
        error "Setup validation failed"
        exit 1
    fi
    
    success "Setup validation completed"
}

# Extract connectivity matrices
extract_matrices() {
    local input_path="$1"
    local output_dir="$2"
    local config_file="$3"
    local atlases="$4"
    local metrics="$5"
    local tracks="$6"
    local pilot="$7"
    local pilot_count="$8"
    local verbose="$9"
    
    log "Starting connectivity matrix extraction..."
    
    # Create organized output directory structure
    local matrices_dir="$output_dir/connectivity_matrices"
    local logs_dir="$output_dir/logs"
    local organized_dir="$output_dir/organized_matrices"
    mkdir -p "$matrices_dir" "$logs_dir" "$organized_dir"
    
    # Build extraction command
    local cmd="python $SCRIPT_DIR/extract_connectivity_matrices.py"
    
    # Add configuration
    if [[ -n "$config_file" ]]; then
        cmd="$cmd --config $config_file"
    fi
    
    # Add custom parameters
    if [[ -n "$atlases" ]]; then
        cmd="$cmd --atlases $atlases"
    fi
    
    if [[ -n "$metrics" ]]; then
        cmd="$cmd --connectivity-values $metrics"
    fi
    
    if [[ -n "$tracks" ]]; then
        cmd="$cmd --track-count $tracks"
    fi
    
    # Add pilot options
    if [[ "$pilot" == "true" ]]; then
        cmd="$cmd --pilot"
        if [[ -n "$pilot_count" ]]; then
            cmd="$cmd --pilot-count $pilot_count"
        fi
    fi
    
    # Add batch processing if input is directory
    if [[ -d "$input_path" ]]; then
        cmd="$cmd --batch"
    fi
    
    # Add verbose flag
    if [[ "$verbose" == "true" ]]; then
        cmd="$cmd --verbose"
    fi
    
    # Add input and output
    cmd="$cmd $input_path $matrices_dir"
    
    # Generate timestamped log file
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$logs_dir/extraction_$timestamp.log"
    
    log "Extraction command: $cmd"
    log "Log file: $log_file"
    
    # Run extraction with logging
    if eval "$cmd 2>&1 | tee $log_file"; then
        success "Connectivity matrix extraction completed successfully"
        
        # Organize the output matrices
        set +e  # Temporarily disable exit on error for organization
        organize_matrices "$matrices_dir" "$organized_dir" "$timestamp"
        local organize_exit_code=$?
        set -e  # Re-enable exit on error
        
        if [[ $organize_exit_code -ne 0 ]]; then
            error "Matrix organization failed with exit code $organize_exit_code"
            exit 1
        fi
        
        # Generate summary
        generate_summary "$output_dir" "$timestamp" "$input_path" "$cmd" "$organized_dir"
    else
        error "Connectivity matrix extraction failed"
        error "Check log file: $log_file"
        exit 1
    fi
}

# Organize extracted matrices into a clean structure
organize_matrices() {
    local raw_dir="$1"
    local organized_dir="$2"
    local timestamp="$3"
    
    log "Organizing connectivity matrices..."
    
    # The DSI Studio output is already organized by atlas in the by_atlas subdirectories
    # We just need to create a cleaner structure
    
    if [[ ! -d "$raw_dir" ]]; then
        warn "No raw directory found to organize: $raw_dir"
        return
    fi
    
    # Find all by_atlas directories
    local atlas_dirs=$(find "$raw_dir" -type d -name "by_atlas" 2>/dev/null)
    
    if [[ -z "$atlas_dirs" ]]; then
        warn "No by_atlas directories found in $raw_dir"
        return
    fi
    
    local total_matrices=0
    local subject_count=0
    local atlas_count=0
    local processed_subjects=()  # Array to track processed subjects
    
    # Create organized structure: organized_matrices/atlas/metric/
    local atlas_base_dirs=($(find "$raw_dir" -type d -name "by_atlas" 2>/dev/null))
    
    for atlas_base_dir in "${atlas_base_dirs[@]}"; do
        if [[ -z "$atlas_base_dir" || ! -d "$atlas_base_dir" ]]; then continue; fi
        
        # Get subject info from the path
        local subject_path=$(dirname "$atlas_base_dir")
        local subject_name=$(basename "$(dirname "$subject_path")")
        
        # Process each atlas in this by_atlas directory
        for atlas_dir in "$atlas_base_dir"/*; do
            if [[ ! -d "$atlas_dir" ]]; then continue; fi
            
            local atlas_name=$(basename "$atlas_dir")
            local target_atlas_dir="$organized_dir/$atlas_name"
            mkdir -p "$target_atlas_dir"
            
            log "Processing atlas: $atlas_name for subject: $subject_name"
            
            # Find connectivity matrix files (connectogram and network_measures)
            local connectogram_files=($(find "$atlas_dir" -name "*.connectogram.txt" 2>/dev/null))
            local network_files=($(find "$atlas_dir" -name "*.network_measures.txt" 2>/dev/null))
            local tract2region_files=($(find "$atlas_dir" -name "*.tract2region.txt" 2>/dev/null))
            
            # Process connectogram files (these are the actual connectivity matrices)
            for file in "${connectogram_files[@]}"; do
                if [[ -z "$file" ]]; then continue; fi
                
                local filename=$(basename "$file")
                
                # Extract metric from filename (pattern: ...atlas.metric.pass.connectogram.txt)
                if [[ "$filename" =~ \.([^.]+)\.pass\.connectogram\.txt$ ]]; then
                    local metric="${BASH_REMATCH[1]}"
                    local target_metric_dir="$target_atlas_dir/$metric"
                    mkdir -p "$target_metric_dir"
                    
                    # Create clean filename
                    local clean_filename="${subject_name}_${atlas_name}_${metric}_connectogram.txt"
                    cp "$file" "$target_metric_dir/$clean_filename"
                    ((total_matrices++))
                fi
            done
            
            # Process network measures files
            for file in "${network_files[@]}"; do
                if [[ -z "$file" ]]; then continue; fi
                
                local filename=$(basename "$file")
                
                # Extract metric from filename
                if [[ "$filename" =~ \.([^.]+)\.pass\.network_measures\.txt$ ]]; then
                    local metric="${BASH_REMATCH[1]}"
                    local target_metric_dir="$target_atlas_dir/$metric"
                    mkdir -p "$target_metric_dir"
                    
                    # Create clean filename
                    local clean_filename="${subject_name}_${atlas_name}_${metric}_network_measures.txt"
                    cp "$file" "$target_metric_dir/$clean_filename"
                fi
            done
            
            # Process tract2region files
            for file in "${tract2region_files[@]}"; do
                if [[ -z "$file" ]]; then continue; fi
                
                local filename=$(basename "$file")
                local target_metric_dir="$target_atlas_dir/tract2region"
                mkdir -p "$target_metric_dir"
                
                # Create clean filename
                local clean_filename="${subject_name}_${atlas_name}_tract2region.txt"
                cp "$file" "$target_metric_dir/$clean_filename"
            done
            
            ((atlas_count++))
        done
        
        # Count unique subjects
        if [[ ! " ${processed_subjects[*]} " =~ " ${subject_name} " ]]; then
            processed_subjects+=("$subject_name")
            ((subject_count++))
        fi
        
    done
    
    # Create subject-wise organization
    create_subject_organization "$raw_dir" "$organized_dir" 
    
    # Create a matrix index file
    create_matrix_index "$organized_dir" "$timestamp" "$total_matrices" "$subject_count" "$atlas_count"
    
    success "Matrices organized successfully"
    log "Organized structure created in: $organized_dir"
    log "Atlas-wise organization: $organized_dir/{atlas}/{metric}/"
    log "Subject-wise organization: $organized_dir/by_subject/{subject}/"
    log "Total matrices organized: $total_matrices"
}

# Create subject-wise organization
create_subject_organization() {
    local raw_dir="$1"
    local organized_dir="$2"
    
    local subjects_dir="$organized_dir/by_subject"
    mkdir -p "$subjects_dir"
    
    # Find all subject directories
    local subject_dirs=$(find "$raw_dir" -maxdepth 1 -type d -name "sub-*" 2>/dev/null)
    
    while IFS= read -r subject_dir; do
        if [[ -z "$subject_dir" || ! -d "$subject_dir" ]]; then continue; fi
        
        local subject_name=$(basename "$subject_dir")
        local subject_clean=$(echo "$subject_name" | sed 's/_[0-9]*$//')  # Remove timestamp
        local subject_target_dir="$subjects_dir/$subject_clean"
        mkdir -p "$subject_target_dir"
        
        # Copy all matrix files for this subject
        find "$subject_dir" -name "*.txt" -type f | while IFS= read -r file; do
            if [[ -n "$file" ]]; then
                local filename=$(basename "$file")
                cp "$file" "$subject_target_dir/"
            fi
        done
        
    done <<< "$subject_dirs"
}

# Create a comprehensive matrix index file
create_matrix_index() {
    local organized_dir="$1"
    local timestamp="$2"
    local total_matrices="$3"
    local subject_count="$4"
    local atlas_count="$5"
    
    local index_file="$organized_dir/matrix_index_$timestamp.json"
    local summary_file="$organized_dir/extraction_summary_$timestamp.txt"
    
    log "Creating matrix index..."
    
    # If parameters not provided, count them
    if [[ -z "$total_matrices" ]]; then
        total_matrices=$(find "$organized_dir" -name "*.txt" | wc -l)
    fi
    if [[ -z "$subject_count" ]]; then
        subject_count=$(find "$organized_dir/by_subject" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    fi
    if [[ -z "$atlas_count" ]]; then
        atlas_count=$(find "$organized_dir" -mindepth 1 -maxdepth 1 -type d -not -name "by_subject" | wc -l)
    fi
    
    # Start JSON structure
    echo "{" > "$index_file"
    echo "  \"extraction_info\": {" >> "$index_file"
    echo "    \"timestamp\": \"$timestamp\"," >> "$index_file"
    echo "    \"extraction_date\": \"$(date)\"" >> "$index_file"
    echo "  }," >> "$index_file"
    echo "  \"organization\": {" >> "$index_file"
    echo "    \"atlases_processed\": $atlas_count," >> "$index_file"
    echo "    \"subjects_processed\": $subject_count," >> "$index_file"
    echo "    \"total_matrices\": $total_matrices" >> "$index_file"
    echo "  }," >> "$index_file"
    echo "  \"structure\": {" >> "$index_file"
    echo "    \"by_atlas\": {" >> "$index_file"
    
    # List atlases and their metrics
    local first_atlas=true
    if [[ -d "$organized_dir" ]]; then
        for atlas_dir in "$organized_dir"/*; do
            if [[ -d "$atlas_dir" && "$(basename "$atlas_dir")" != "by_subject" ]]; then
                local atlas_name=$(basename "$atlas_dir")
                
                if [[ "$first_atlas" == "false" ]]; then
                    echo "," >> "$index_file"
                fi
                echo "      \"$atlas_name\": {" >> "$index_file"
                echo "        \"metrics\": [" >> "$index_file"
                
                local first_metric=true
                for metric_dir in "$atlas_dir"/*; do
                    if [[ -d "$metric_dir" ]]; then
                        local metric_name=$(basename "$metric_dir")
                        local matrix_count=$(find "$metric_dir" -name "*.txt" | wc -l)
                        
                        if [[ "$first_metric" == "false" ]]; then
                            echo "," >> "$index_file"
                        fi
                        echo -n "          {\"name\": \"$metric_name\", \"matrices\": $matrix_count}" >> "$index_file"
                        first_metric=false
                    fi
                done
                
                echo "" >> "$index_file"
                echo "        ]" >> "$index_file"
                echo -n "      }" >> "$index_file"
                first_atlas=false
            fi
        done
    fi
    
    echo "" >> "$index_file"
    echo "    }," >> "$index_file"
    echo "    \"by_subject\": {" >> "$index_file"
    echo "      \"subjects\": [" >> "$index_file"
    
    # List subjects
    local first_subject=true
    if [[ -d "$organized_dir/by_subject" ]]; then
        for subject_dir in "$organized_dir/by_subject"/*; do
            if [[ -d "$subject_dir" ]]; then
                local subject_name=$(basename "$subject_dir")
                local subject_matrices=$(find "$subject_dir" -name "*.txt" | wc -l)
                
                if [[ "$first_subject" == "false" ]]; then
                    echo "," >> "$index_file"
                fi
                echo -n "        {\"id\": \"$subject_name\", \"matrices\": $subject_matrices}" >> "$index_file"
                first_subject=false
            fi
        done
    fi
    
    echo "" >> "$index_file"
    echo "      ]" >> "$index_file"
    echo "    }" >> "$index_file"
    echo "  }" >> "$index_file"
    echo "}" >> "$index_file"
    
    # Create human-readable summary
    cat > "$summary_file" << EOF
CONNECTIVITY EXTRACTION SUMMARY
===============================
Generated: $(date)
Timestamp: $timestamp

ORGANIZATION STRUCTURE:
- Atlases processed: $atlas_count
- Subjects processed: $subject_count
- Total matrices: $total_matrices

DIRECTORY STRUCTURE:
organized_matrices/
├── by_atlas/
│   ├── {atlas_name}/
│   │   └── {metric_name}/
│   │       └── {subject}_{atlas}_{metric}_connectogram.txt
│   └── ...
└── by_subject/
    ├── {subject_ID}/
    │   └── all_matrices_for_subject.txt
    └── ...

FILES:
- matrix_index_$timestamp.json - Machine-readable index
- extraction_summary_$timestamp.txt - This human-readable summary

NEXT STEPS:
1. Graph Analysis: ./02_analyze_graphs.sh $organized_dir ../graph_analysis/
2. Quality Control: Check subjects with low matrix counts
3. Atlas Comparison: Compare metrics across different atlases
EOF
    
    success "Matrix index created: $index_file"
    success "Summary created: $summary_file"
}

# Generate extraction summary
generate_summary() {
    local output_dir="$1"
    local timestamp="$2"
    local input_path="$3"
    local command="$4"
    local organized_dir="$5"
    
    local summary_file="$output_dir/summary_$timestamp.json"
    local matrices_dir="$output_dir/connectivity_matrices"
    
    log "Generating extraction summary..."
    
    # Count extracted files from organized directory
    local total_matrices=0
    local subject_count=0
    local atlas_count=0
    
    if [[ -d "$organized_dir" ]]; then
        total_matrices=$(find "$organized_dir" -name "*.txt" | wc -l)
        atlas_count=$(find "$organized_dir" -mindepth 1 -maxdepth 1 -type d -not -name "by_subject" | wc -l)
        if [[ -d "$organized_dir/by_subject" ]]; then
            subject_count=$(find "$organized_dir/by_subject" -mindepth 1 -maxdepth 1 -type d | wc -l)
        fi
    fi
    
    # Create summary JSON
    # Create detailed JSON metadata for pipeline integration
    pipeline_metadata="$organized_dir/pipeline_metadata_$timestamp.json"
    
    cat > "$pipeline_metadata" << EOF
{
    "pipeline_metadata": {
        "step": "01_extract_connectome",
        "timestamp": "$timestamp",
        "version": "1.0",
        "input_configuration": "$config_file",
        "processing_details": {
            "input_path": "$input_path",
            "output_directory": "$output_dir",
            "track_count": $(python -c "
import json
with open('$config_file', 'r') as f:
    config = json.load(f)
print(config.get('tract_count', config.get('track_count', 4000)))
")
        },
        "results": {
            "total_matrices_extracted": $total_matrices,
            "subjects_processed": $subject_count,
            "atlases_processed": $atlas_count,
            "raw_matrices_directory": "$matrices_dir",
            "organized_matrices_directory": "$organized_dir",
            "network_measures_available": true
        },
        "file_structure": {
            "connectivity_matrices": {
                "by_atlas": "$organized_dir/{atlas}/{metric}/",
                "by_subject": "$organized_dir/by_subject/{subject}/",
                "pattern": "*.connectivity.mat"
            },
            "network_measures": {
                "location": "$organized_dir/by_atlas/{atlas}/",
                "pattern": "*.network_measures.txt",
                "description": "Pre-computed graph theory metrics from DSI Studio"
            },
            "other_outputs": {
                "connectograms": "*.connectogram.txt",
                "statistics": "*.stat.txt"
            }
        },
        "atlas_mapping": $(python -c "
import json
import os
from pathlib import Path

atlases = []
organized_dir = '$organized_dir'
by_atlas_dir = os.path.join(organized_dir, 'by_atlas')

if os.path.exists(by_atlas_dir):
    for atlas_dir in os.listdir(by_atlas_dir):
        atlas_path = os.path.join(by_atlas_dir, atlas_dir)
        if os.path.isdir(atlas_path):
            # Count network measures files for this atlas
            network_measures_count = len([f for f in os.listdir(atlas_path) if f.endswith('network_measures.txt')])
            atlases.append({
                'name': atlas_dir,
                'path': atlas_path,
                'network_measures_files': network_measures_count
            })

print(json.dumps(atlases, indent=2))
"),
        "connectivity_metrics": $(python -c "
import json
with open('$config_file', 'r') as f:
    config = json.load(f)
metrics = config.get('connectivity_metrics', [])
print(json.dumps(metrics))
"),
        "ready_for_next_steps": {
            "02_compute_graph_metrics": {
                "input_data": "$pipeline_metadata",
                "ready": true,
                "description": "Network measures are pre-computed and ready for collection"
            },
            "03_optimize_metrics": {
                "ready": false,
                "requires": "Output from step 02"
            }
        },
        "logs": {
            "extraction_log": "$output_dir/logs/extraction_$timestamp.log",
            "organization_log": "$organized_dir/organization_$timestamp.log"
        }
    }
}
EOF

    success "Pipeline metadata saved to: $pipeline_metadata"
    
    # Create traditional summary for backward compatibility
    cat > "$summary_file" << EOF
{
    "extraction_summary": {
        "timestamp": "$timestamp",
        "input_path": "$input_path",
        "output_directory": "$output_dir",
        "command_executed": "$command",
        "results": {
            "total_matrices_extracted": $total_matrices,
            "subjects_processed": $subject_count,
            "atlases_processed": $atlas_count,
            "raw_matrices_directory": "$matrices_dir",
            "organized_matrices_directory": "$organized_dir"
        },
        "organization": {
            "by_atlas": "$organized_dir/{atlas}/{metric}/",
            "by_subject": "$organized_dir/by_subject/{subject}/",
            "index_file": "$organized_dir/matrix_index_$timestamp.json",
            "summary_file": "$organized_dir/extraction_summary_$timestamp.txt"
        },
        "next_steps": [
            "02_compute_graph_metrics.sh - Use pipeline_metadata_$timestamp.json as input",
            "03_optimize_metrics.sh - Optimize atlas/metric combinations",
            "04_statistical_analysis.sh - Run statistical models"
        ],
        "pipeline_integration": {
            "metadata_file": "$pipeline_metadata",
            "use_for_step_02": "pipeline_metadata_$timestamp.json"
        }
    }
}
EOF
    
    # Display summary
    echo
    echo "============================================================================"
    echo "                        EXTRACTION SUMMARY"
    echo "============================================================================"
    echo "Timestamp:           $timestamp"
    echo "Input:               $input_path"
    echo "Output Directory:    $output_dir"
    echo "Matrices Extracted:  $total_matrices"
    echo "Subjects Processed:  $subject_count"
    echo "Atlases Processed:   $atlas_count"
    echo "Raw Data:            $matrices_dir"
    echo "Organized Data:      $organized_dir"
    echo "Index File:          $organized_dir/matrix_index_$timestamp.json"
    echo "Pipeline Metadata:   $pipeline_metadata"
    echo "Log File:            $output_dir/logs/extraction_$timestamp.log"
    echo "Summary File:        $summary_file"
    echo "============================================================================"
    echo
    echo "ORGANIZED STRUCTURE:"
    echo "  By Atlas:    $organized_dir/{atlas}/{metric}/"
    echo "  By Subject:  $organized_dir/by_subject/{subject}/"
    echo
    echo "Next Step:"
    echo "  ./02_analyze_graphs.sh $organized_dir $output_dir"
    echo
}

# Main function
main() {
    local input_path=""
    local output_dir=""
    local config_file=""
    local atlases=""
    local metrics=""
    local tracks=""
    local pilot="false"
    local pilot_count="2"
    local validate="false"
    local organize_only="false"
    local verbose="false"
    
    local sweep="false"
    local quick_sweep="false"
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            -a|--atlases)
                atlases="$2"
                shift 2
                ;;
            -m|--metrics)
                metrics="$2"
                shift 2
                ;;
            -t|--tracks)
                tracks="$2"
                shift 2
                ;;
            -p|--pilot)
                pilot="true"
                shift
                ;;
            --pilot-count)
                pilot_count="$2"
                shift 2
                ;;
            --validate)
                validate="true"
                shift
                ;;
            --organize-only)
                organize_only="true"
                shift
                ;;
            -v|--verbose)
                verbose="true"
                shift
                ;;
            --sweep)
                sweep="true"
                shift
                ;;
            --quick-sweep)
                sweep="true"
                quick_sweep="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$input_path" ]]; then
                    input_path="$1"
                elif [[ -z "$output_dir" ]]; then
                    output_dir="$1"
                else
                    error "Too many arguments"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Sweep-Modus: Parameter-Sweep für einen Piloten
    if [[ "$sweep" == "true" ]]; then
        echo "[SWEEP] Running parameter sweep mode..."
        
        # Use default config if not specified
        if [[ -z "$config_file" ]]; then
            config_file="$DEFAULT_CONFIG"
        fi
        
        # Validate config file
        if [[ -n "$config_file" ]] && [[ ! -f "$config_file" ]]; then
            error "Configuration file not found: $config_file"
            exit 1
        fi
        
        echo "[SWEEP] Using config: $config_file"
        
        # Schreibe temporäres Python-Sweep-Skript
        SWEEP_SCRIPT="sweep_tmp.py"
        cat > "$SWEEP_SCRIPT" << 'EOF'
import itertools
import json
import os
import random
import subprocess
import numpy as np
def get_subjects(data_dir):
    return [f for f in os.listdir(data_dir) if f.startswith('sub-')]
def count_zeros_in_matrix(matrix_path):
    # DSI Studio matrices haben 2 Header-Zeilen
    try:
        mat = np.loadtxt(matrix_path, skiprows=2)
        # Nur die numerischen Spalten verwenden (ab Spalte 2)
        if mat.ndim == 2 and mat.shape[1] > 2:
            numeric_data = mat[:, 2:]  # Erste 2 Spalten sind Labels
        else:
            numeric_data = mat
        return int((numeric_data == 0).sum()), int(numeric_data.size)
    except Exception as e:
        # Fallback: Versuche anders zu parsen
        with open(matrix_path, 'r') as f:
            lines = f.readlines()
        # Finde die erste numerische Zeile
        numeric_lines = []
        for i, line in enumerate(lines):
            if i < 2:  # Überspringe Header
                continue
            parts = line.strip().split()
            if len(parts) > 2:
                try:
                    # Versuche die numerischen Teile zu konvertieren (ab Index 2)
                    numeric_parts = [float(x) for x in parts[2:]]
                    numeric_lines.append(numeric_parts)
                except ValueError:
                    continue
        if numeric_lines:
            mat = np.array(numeric_lines)
            return int((mat == 0).sum()), int(mat.size)
        else:
            raise Exception(f"Could not parse matrix: {e}")
def run_sweep(config_path, data_dir, results_dir, sweep_log='sweep_results.csv', quick=False):
    # Lade Sweep-Parameter aus der Konfigurationsdatei
    with open(config_path) as f:
        base_config = json.load(f)
    
    # Hole Sweep-Parameter aus der Config oder verwende Defaults
    sweep_config = base_config.get('sweep_parameters', {})
    
    if quick and 'quick_sweep' in sweep_config:
        # Verwende quick_sweep Parameter
        sweep_params = sweep_config['quick_sweep']
        print(f"[SWEEP] Using QUICK sweep parameters")
    else:
        # Verwende normale Sweep Parameter
        sweep_params = sweep_config
        print(f"[SWEEP] Using FULL sweep parameters")
    
    # Parameter-Ranges aus Config laden
    otsu_range = sweep_params.get('otsu_range', [0.3, 0.4])
    min_length_range = sweep_params.get('min_length_range', [10, 20])
    track_voxel_ratio_range = sweep_params.get('track_voxel_ratio_range', [2.0])
    fa_threshold_range = sweep_params.get('fa_threshold_range', [0.1])
    sweep_tract_count = sweep_params.get('sweep_tract_count', 50000)
    
    print(f"[SWEEP] Parameter ranges from config:")
    print(f"  otsu_range: {otsu_range}")
    print(f"  min_length_range: {min_length_range}")
    print(f"  track_voxel_ratio_range: {track_voxel_ratio_range}")
    print(f"  fa_threshold_range: {fa_threshold_range}")
    print(f"  sweep_tract_count: {sweep_tract_count}")
    
    subjects = get_subjects(data_dir)
    pilot = random.choice(subjects)
    pilot_file = os.path.join(data_dir, pilot)
    print(f"Pilot subject: {pilot}")
    print(f"Pilot file: {pilot_file}")
    
    with open(config_path) as f:
        base_config = json.load(f)
    
    with open(sweep_log, 'w') as logf:
        logf.write('otsu,min_length,track_voxel_ratio,fa_threshold,zeros,total,percent_zeros,combination,status\n')
    
    combination_num = 0
    total_combinations = len(list(itertools.product(otsu_range, min_length_range, track_voxel_ratio_range, fa_threshold_range)))
    
    for otsu, min_len, tvr, fa in itertools.product(otsu_range, min_length_range, track_voxel_ratio_range, fa_threshold_range):
        combination_num += 1
        print(f"\\n[SWEEP] Testing combination {combination_num}/{total_combinations}: otsu={otsu}, min_len={min_len}, tvr={tvr}, fa={fa}")
        
        # Cleanup previous temp files
        for tmp_file in ['tmp_sweep_config*.json']:
            import glob
            for f in glob.glob(tmp_file):
                try:
                    os.remove(f)
                except:
                    pass
        
        config = json.loads(json.dumps(base_config))
        config['tracking_parameters']['otsu_threshold'] = otsu
        config['tracking_parameters']['min_length'] = min_len
        config['tracking_parameters']['track_voxel_ratio'] = tvr
        config['tracking_parameters']['fa_threshold'] = fa
        
        # Reduzierte Anzahl Trakte um Crashes zu vermeiden
        config['tract_count'] = sweep_tract_count  # Verwende die einstellbare Variable
        
        tmp_config = f'tmp_sweep_config_{combination_num}.json'
        with open(tmp_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Führe direkt die Python-Extraktion aus mit Error Handling
        sweep_output_dir = os.path.join(results_dir, f'sweep_combo_{combination_num}')
        os.makedirs(sweep_output_dir, exist_ok=True)
        
        cmd = [
            'python3', '/Volumes/Work/github/braingraph-pipeline/extract_connectivity_matrices.py',
            '--config', tmp_config,
            pilot_file,
            os.path.join(sweep_output_dir, 'connectivity_matrices')
        ]
        
        print(f"[SWEEP] Running: {' '.join(cmd)}")
        status = "SUCCESS"
        
        try:
            # Längeres Timeout und besseres Error Handling
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[SWEEP] Error in extraction (return code {result.returncode}): {result.stderr}")
                status = "EXTRACTION_ERROR"
                zeros, total, percent_zeros = 'EXTRACTION_ERROR', 'EXTRACTION_ERROR', 'EXTRACTION_ERROR'
            else:
                print(f"[SWEEP] Extraction successful")
                
                # Suche nach Matrizen in der rohen Output-Struktur (ohne Organisation)
                raw_dir = os.path.join(sweep_output_dir, 'connectivity_matrices')
                
                # Suche nach connectogram-Dateien
                import glob
                pattern = os.path.join(raw_dir, '**', '*count.pass.connectogram.txt')
                files = glob.glob(pattern, recursive=True)
                
                if files:
                    matrix_path = files[0]
                    try:
                        zeros, total = count_zeros_in_matrix(matrix_path)
                        percent_zeros = zeros / total * 100
                        print(f"[SWEEP] Matrix analysis: {zeros}/{total} zeros ({percent_zeros:.1f}%)")
                        status = "SUCCESS"
                    except Exception as e:
                        print(f"[SWEEP] Error analyzing matrix: {e}")
                        zeros, total, percent_zeros = 'ANALYSIS_ERROR', 'ANALYSIS_ERROR', 'ANALYSIS_ERROR'
                        status = "ANALYSIS_ERROR"
                else:
                    print(f"[SWEEP] No matrix found at {pattern}")
                    # Versuche andere Dateitypen zu finden
                    all_files = glob.glob(os.path.join(raw_dir, '**', '*.txt'), recursive=True)
                    print(f"[SWEEP] Found {len(all_files)} files in output directory")
                    if all_files:
                        print(f"[SWEEP] Example files: {all_files[:3]}")
                    zeros, total, percent_zeros = 'NO_MATRIX', 'NO_MATRIX', 'NO_MATRIX'
                    status = "NO_MATRIX"
                    
        except subprocess.TimeoutExpired:
            print(f"[SWEEP] Timeout for combination {combination_num}")
            zeros, total, percent_zeros = 'TIMEOUT', 'TIMEOUT', 'TIMEOUT'
            status = "TIMEOUT"
        except Exception as e:
            print(f"[SWEEP] Exception: {e}")
            zeros, total, percent_zeros = 'EXCEPTION', 'EXCEPTION', 'EXCEPTION'
            status = "EXCEPTION"
        
        # Log das Ergebnis
        with open(sweep_log, 'a') as logf:
            logf.write(f"{otsu},{min_len},{tvr},{fa},{zeros},{total},{percent_zeros},{combination_num},{status}\\n")
        
        # Cleanup
        if os.path.exists(tmp_config):
            os.remove(tmp_config)
            
        # Kurze Pause zwischen den Kombinationen um DSI Studio zu entlasten
        import time
        time.sleep(2)
        
    print(f"\\n[SWEEP] Completed {combination_num} combinations. Results in {sweep_log}")
if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='01_working_config.json')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--quick', action='store_true', help='Use quick sweep parameters')
    args = parser.parse_args()
    run_sweep(args.config, args.data_dir, args.results_dir, quick=args.quick)
EOF
        # Führe Sweep aus
        if [[ "$quick_sweep" == "true" ]]; then
            python3 "$SWEEP_SCRIPT" --config "$config_file" --data_dir "$input_path" --results_dir "$output_dir" --quick
        else
            python3 "$SWEEP_SCRIPT" --config "$config_file" --data_dir "$input_path" --results_dir "$output_dir"
        fi
        echo "[SWEEP] Sweep finished. Results in sweep_results.csv."
        rm -f "$SWEEP_SCRIPT" tmp_sweep_config.json
        exit 0
    fi
    # Validate required arguments
    if [[ -z "$input_path" ]] || [[ -z "$output_dir" ]]; then
        error "Missing required arguments"
        show_usage
        exit 1
    fi
    
    # Validate input path
    if [[ ! -e "$input_path" ]]; then
        error "Input path does not exist: $input_path"
        exit 1
    fi
    
    # Use default config if not specified
    if [[ -z "$config_file" ]]; then
        config_file="$DEFAULT_CONFIG"
    fi
    
    # Validate config file
    if [[ -n "$config_file" ]] && [[ ! -f "$config_file" ]]; then
        error "Configuration file not found: $config_file"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    echo "============================================================================"
    echo "                   DSI STUDIO CONNECTIVITY EXTRACTION"
    echo "============================================================================"
    echo "Input:          $input_path"
    echo "Output:         $output_dir"
    echo "Config:         $config_file"
    echo "Atlases:        ${atlases:-$DEFAULT_ATLASES}"
    echo "Metrics:        ${metrics:-$DEFAULT_METRICS}"
    echo "Tracks:         ${tracks:-$DEFAULT_TRACKS}"
    echo "Pilot Mode:     $pilot"
    if [[ "$pilot" == "true" ]]; then
        echo "Pilot Count:    $pilot_count"
    fi
    echo "Validate:       $validate"
    echo "============================================================================"
    echo
    
    # Check and activate environment
    check_environment
    activate_environment
    
    # Validate configuration file
    validate_config "$config_file"
    
    # Validate DSI Studio tools
    validate_dsi_tools
    
    # Run validation if requested
    if [[ "$validate" == "true" ]]; then
        run_validation "$config_file" "$input_path"
    fi
    
    # Handle organize-only mode or full extraction
    if [[ "$organize_only" == "true" ]]; then
        log "Running in organize-only mode..."
        
        # Set up directories
        local matrices_dir="$output_dir/connectivity_matrices"
        local organized_dir="$output_dir/organized_matrices"
        local timestamp=$(date +%Y%m%d_%H%M%S)
        
        # Ensure output directories exist
        mkdir -p "$organized_dir"
        mkdir -p "$output_dir/logs"
        
        # Check if raw matrices exist
        if [[ ! -d "$matrices_dir" ]]; then
            error "No connectivity matrices found at: $matrices_dir"
            error "Run without --organize-only to extract matrices first."
            exit 1
        fi
        
        # Organize the matrices
        set +e  # Temporarily disable exit on error for organization
        organize_matrices "$matrices_dir" "$organized_dir" "$timestamp"
        local organize_exit_code=$?
        set -e  # Re-enable exit on error
        
        if [[ $organize_exit_code -ne 0 ]]; then
            error "Matrix organization failed with exit code $organize_exit_code"
            exit 1
        fi
        
        # Generate summary
        generate_summary "$output_dir" "$timestamp" "$input_path" "organize-only" "$organized_dir"
    else
        # Extract connectivity matrices (includes organization)
        extract_matrices "$input_path" "$output_dir" "$config_file" "$atlases" "$metrics" "$tracks" "$pilot" "$pilot_count" "$verbose"
    fi
    
    success "Step 01 completed successfully!"
    echo ""
    echo "🚀 Next Steps:"
    echo "  Step 02: ./02_compute_graph_metrics.sh --input <pipeline_metadata.json>"
    echo "  Step 03: ./03_optimize_metrics.sh <metrics.csv>"
    echo ""
    echo "💡 Pipeline metadata available for automatic step chaining!"
}

# Run main function with all arguments
main "$@"
