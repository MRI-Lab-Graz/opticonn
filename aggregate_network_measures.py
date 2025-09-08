#!/usr/bin/env python3
"""
Aggregate individual network measures CSV files into a single consolidated CSV file.
This script collects all *network_measures.csv files from the organized matrices directory
and combines them into one file suitable for input to Step 02 (metric_optimizer.py).
"""

import os
import sys
import pandas as pd
import glob
from pathlib import Path

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python aggregate_network_measures.py <input_dir> <output_file>")
        print("Example: python aggregate_network_measures.py test_results/organized_matrices test_results/aggregated_network_measures.csv")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    success = aggregate_network_measures(input_dir, output_file)
    sys.exit(0 if success else 1)
