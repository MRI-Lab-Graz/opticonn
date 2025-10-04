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
    # Find all network_measures.csv files
    pattern = os.path.join(input_dir, "**", "*network_measures.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        print(f"No network_measures.csv files found in {input_dir}")
        return False
    
    print(f"Found {len(csv_files)} network_measures.csv files")
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            # Extract metadata from path
            path_parts = Path(csv_file).parts
            
            # Find subject ID and other metadata from path
            subject_id = None
            atlas = None
            metric_type = None
            
            for part in path_parts:
                if part.startswith("sub-"):
                    # Extract just the subject part before the timestamp
                    subject_id = part.split(".odf.qsdr")[0]
            
            # NEW: Extract atlas from the new results/atlas_name/ structure
            if "results" in path_parts:
                results_idx = list(path_parts).index("results")
                if results_idx + 1 < len(path_parts):
                    # The next directory after results/ is the atlas name
                    atlas = path_parts[results_idx + 1]
            
            # LEGACY: Support old by_atlas structure for backward compatibility
            elif "by_atlas" in path_parts:
                atlas_idx = list(path_parts).index("by_atlas")
                if atlas_idx + 1 < len(path_parts):
                    atlas_part = path_parts[atlas_idx + 1]
                    # Extract atlas name (e.g., "HCP-MMP" from "HCP-MMP.tt.gz.HCP-MMP.count..pass.network_measures.csv")
                    if "." in atlas_part:
                        atlas = atlas_part.split(".")[0]
                    else:
                        atlas = atlas_part
            
            # Extract metric type from filename
            filename = Path(csv_file).name
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
            
            # Read the CSV file (it's actually tab-separated with no header)
            # Only read the first part until we hit "network_measures" which indicates the connectivity matrix
            lines = []
            with open(csv_file, 'r') as f:
                for line in f:
                    if line.startswith('network_measures'):
                        break
                    lines.append(line.strip())
            
            if not lines:
                continue
                
            # Parse the network measures
            row_data = {'subject_id': subject_id, 'atlas': atlas, 'connectivity_metric': metric_type}
            for line in lines:
                if '\t' in line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        metric_name = parts[0].strip()
                        try:
                            metric_value = float(parts[1].strip())
                            row_data[metric_name] = metric_value
                        except ValueError:
                            continue
            
            all_data.append(row_data)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if not all_data:
        print("No data could be processed")
        return False
    
    # Create consolidated DataFrame
    result_df = pd.DataFrame(all_data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    
    print(f"Aggregated data saved to: {output_file}")
    print(f"Shape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)}")
    
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
