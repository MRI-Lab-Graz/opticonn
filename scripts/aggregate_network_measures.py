#!/usr/bin/env python3
"""
Aggregate individual network measures CSV files into a single consolidated CSV file.
This script collects all *network_measures.csv files from the organized matrices directory
and combines them into one file suitable for input to Step 02 (metric_optimizer.py).
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import glob
from pathlib import Path

from scripts.utils.runtime import configure_stdio


def aggregate_network_measures(input_dir, output_file):
    """
    Aggregate network measures from individual CSV files into one consolidated file.

    Args:
        input_dir (str): Directory containing organized matrices with network_measures.csv files
        output_file (str): Path to output aggregated CSV file
    """
    # Find all network_measures.csv files (primary format)
    pattern = os.path.join(input_dir, "**", "*network_measures.csv")
    csv_files = glob.glob(pattern, recursive=True)

    # If no network_measures.csv files found, look for connectivity CSVs in nested structure
    # This handles the sweep output format from extract_connectivity_matrices.py
    if not csv_files:
        # Look for connectivity matrices in the results/ subdirectories
        pattern_conn = os.path.join(input_dir, "**", "results", "**", "*.connectivity.csv")
        csv_files = glob.glob(pattern_conn, recursive=True)
        
        if csv_files:
            print(f"No network_measures.csv files found. Using {len(csv_files)} connectivity CSV files from nested structure")
        else:
            print(f"No network_measures.csv or connectivity CSV files found in {input_dir}")
            return False
    else:
        print(f"Found {len(csv_files)} network_measures.csv files")

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return False

    print(f"Processing {len(csv_files)} CSV files")

    all_data = []

    for csv_file in csv_files:
        try:
            # Extract metadata from path
            path_parts = Path(csv_file).parts
            filename = Path(csv_file).name

            # Find subject ID and other metadata from path
            subject_id = None
            atlas = None
            metric_type = None

            # Extract subject ID from path (support multiple formats)
            for i, part in enumerate(path_parts):
                if part.startswith("sub-"):
                    subject_id = part.split(".odf.qsdr")[0]
                elif part.startswith("P0") or part.startswith("P1"):
                    # Handle P0XX format (e.g., P040_105)
                    subject_id = part.split("_")[0] if "_" in part else part
                    break

            # Extract atlas from the new results/atlas_name/ structure
            if "results" in path_parts:
                results_indices = [i for i, part in enumerate(path_parts) if part == "results"]
                if results_indices:
                    results_idx = results_indices[-1]
                    if results_idx + 1 < len(path_parts):
                        atlas = path_parts[results_idx + 1]

            # LEGACY: Support old by_atlas structure for backward compatibility
            elif "by_atlas" in path_parts:
                atlas_idx = list(path_parts).index("by_atlas")
                if atlas_idx + 1 < len(path_parts):
                    atlas_part = path_parts[atlas_idx + 1]
                    if "." in atlas_part:
                        atlas = atlas_part.split(".")[0]
                    else:
                        atlas = atlas_part

            # Extract metric type from filename
            if ".count." in filename:
                metric_type = "count"
            elif ".fa." in filename:
                metric_type = "fa"
            elif ".qa." in filename:
                metric_type = "qa"
            elif ".ncount2." in filename:
                metric_type = "ncount2"
            else:
                metric_type = "unknown"

            # Initialize row data with required columns for metric_optimizer.py
            row_data = {
                'subject_id': subject_id or "unknown",
                'atlas': atlas or "unknown",
                'connectivity_metric': metric_type
            }

            # Read the CSV file - format can be:
            # 1. network_measures.csv: TAB-separated key-value pairs (metric_name \t value)
            # 2. connectivity.csv: Pandas DataFrame with regions as index/columns
            
            try:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                
                # Check if this is network_measures.csv format (tab-separated key-value pairs)
                is_network_measures = False
                for line in lines[:5]:  # Check first few lines
                    if '\t' in line and not line.strip().startswith('network_measures'):
                        is_network_measures = True
                        break
                
                if is_network_measures:
                    # Parse network_measures.csv format: metric_name \t value
                    for line in lines:
                        if line.startswith('network_measures'):
                            break  # Stop at the per-node measures section
                        line = line.strip()
                        if '\t' in line:
                            parts = line.split('\t', 1)  # Split on first tab only
                            if len(parts) == 2:
                                metric_name = parts[0].strip()
                                try:
                                    metric_value = float(parts[1].strip())
                                    row_data[metric_name] = metric_value
                                except (ValueError, IndexError):
                                    continue
                else:
                    # This is a connectivity matrix CSV - read with pandas and extract statistics
                    df = pd.read_csv(csv_file, index_col=0)  # First column is index (region names)
                    matrix = df.values
                    
                    # Compute network statistics from connectivity matrix
                    # Skip NaN values that might be in the matrix
                    matrix_clean = np.where(np.isnan(matrix), 0, matrix)
                    row_data['connection_count'] = float(np.sum(matrix_clean > 0))
                    row_data['mean_weight'] = float(np.mean(matrix_clean))
                    row_data['sum_weight'] = float(np.sum(matrix_clean))
                    row_data['density'] = float(np.sum(matrix_clean > 0) / matrix_clean.size) if matrix_clean.size > 0 else 0.0
            except Exception as parse_error:
                # If parsing fails, add placeholder metrics so row still contributes grouping key
                print(f"Warning: Could not parse {csv_file}: {parse_error}")
                row_data['density'] = 0.0

            all_data.append(row_data)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    if not all_data:
        print("No data could be processed")
        return False

    # Create consolidated DataFrame from all per-subject records
    all_records_df = pd.DataFrame(all_data)
    print(f"Loaded {len(all_records_df)} per-subject records")

    # ===== CRITICAL AGGREGATION STEP =====
    # Group by atlas and connectivity_metric, then aggregate network properties across subjects
    # This produces ONE row per (atlas, metric) combination with statistics from all subjects
    
    groupby_cols = ['atlas', 'connectivity_metric']
    agg_dict = {}
    
    # Identify all metric columns (exclude grouping columns and subject_id)
    metric_cols = [col for col in all_records_df.columns 
                   if col not in groupby_cols + ['subject_id']]
    
    # For each metric column, compute mean, std, min, max across subjects
    for col in metric_cols:
        agg_dict[col] = ['mean', 'std', 'min', 'max', 'count']
    
    # Perform grouped aggregation
    if len(all_records_df) > 0:
        result_df = all_records_df.groupby(groupby_cols).agg(agg_dict)
        
        # Flatten multi-level column names (e.g., ('density', 'mean') -> 'density_mean')
        result_df.columns = ['_'.join(col).strip('_') for col in result_df.columns.values]
        result_df = result_df.reset_index()
        
        print(f"Aggregated to {len(result_df)} atlas/metric combinations")
    else:
        # Fallback if aggregation is empty: use raw records
        result_df = all_records_df

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)

    print(f"✅ Aggregated data saved to: {output_file}")
    print(f"   Shape: {result_df.shape}")
    print(f"   Columns: {list(result_df.columns)}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-subject network measures into a consolidated CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Organized matrices directory containing network_measures.csv files")
    parser.add_argument("output_file", help="Destination for aggregated CSV")
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        default=None,
        help="Disable emoji in console output (useful for limited terminals)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a safe dry-run: list/count files that would be processed without writing output",
    )

    # If no args provided, print help (to comply with global instructions)
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    # If dry-run requested, do a safe preview and exit
    if args.dry_run:
        pattern = os.path.join(args.input_dir, "**", "*network_measures.csv")
        csv_files = glob.glob(pattern, recursive=True)
        print("[DRY-RUN] Aggregate network measures preview")
        print(f"[DRY-RUN] Input directory: {args.input_dir}")
        print(f"[DRY-RUN] Output file (would be): {args.output_file}")
        print(f"[DRY-RUN] Found {len(csv_files)} matching files")
        if csv_files:
            print("[DRY-RUN] First 5 files:")
            for f in csv_files[:5]:
                print(f"  - {f}")
        return 0

    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return 1

    success = aggregate_network_measures(args.input_dir, args.output_file)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
