#!/bin/bash
# Simple script to organize already extracted matrices

# Source the functions from the main script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import functions from the main script
source <(grep -A 200 "^organize_matrices()" "$SCRIPT_DIR/01_extract_connectome.sh" | head -200)
source <(grep -A 50 "^create_subject_organization()" "$SCRIPT_DIR/01_extract_connectome.sh" | head -50)
source <(grep -A 50 "^create_matrix_index()" "$SCRIPT_DIR/01_extract_connectome.sh" | head -50)

# Set up logging functions
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - $1"; }
success() { echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS - $1"; }
warn() { echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING - $1"; }
error() { echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR - $1"; }

# Set directories
matrices_dir="./results/connectivity_matrices"
organized_dir="./results/organized_matrices"
timestamp="20250808_124134"

# Clean up the partial organized_matrices directory
if [[ -d "$organized_dir" ]]; then
    log "Cleaning up existing organized_matrices directory..."
    rm -rf "$organized_dir"
fi

mkdir -p "$organized_dir"

# Run organization
log "Starting matrix organization..."
organize_matrices "$matrices_dir" "$organized_dir" "$timestamp"

success "Matrix organization completed!"
