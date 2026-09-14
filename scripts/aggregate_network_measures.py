# Vendored aggregate_network_measures (adapted)

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
import glob
from pathlib import Path

from scripts.utils.runtime import configure_stdio


def aggregate_network_measures(input_dir, output_file):
    pattern = os.path.join(input_dir, "**", "*network_measures.csv")
    csv_files = glob.glob(pattern, recursive=True)
    if not csv_files:
        print(f"No network_measures.csv files found in {input_dir}")
        return False
    print(f"Found {len(csv_files)} network_measures.csv files")
    all_data = []
    for csv_file in csv_files:
        try:
            subject_id = Path(csv_file).stem
            metric_type = "unknown"
            filename = Path(csv_file).name
            if ".count." in filename:
                metric_type = "count"
            row_data = {"subject_id": subject_id, "connectivity_metric": metric_type}
            # Read first lines until network_measures
            lines = []
            with open(csv_file, "r") as f:
                for line in f:
                    if line.startswith("network_measures"):
                        break
                    lines.append(line.strip())
            if not lines:
                continue
            for line in lines:
                if "\t" in line:
                    parts = line.split("\t")
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
    result_df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Aggregated data saved to: {output_file}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-subject network measures into a consolidated CSV"
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_file")
    parser.add_argument("--no-emoji", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()
    configure_stdio(args.no_emoji)
    if args.dry_run:
        pattern = os.path.join(args.input_dir, "**", "*network_measures.csv")
        csv_files = glob.glob(pattern, recursive=True)
        print("[DRY-RUN] Aggregate network measures preview")
        print(f"[DRY-RUN] Found {len(csv_files)} matching files")
        return 0
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return 1
    success = aggregate_network_measures(args.input_dir, args.output_file)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
