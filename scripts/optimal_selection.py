# Vendored optimal_selection (light adaptation)

from __future__ import annotations

import pandas as pd
import json
from pathlib import Path
from scripts.utils.runtime import configure_stdio


def main():
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Select optimal atlas/metric combinations"
    )
    parser.add_argument(
        "optimization_file", nargs="?", help="CSV file with optimization results"
    )
    parser.add_argument("output_dir", nargs="?", help="Output dir")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        help="CSV input file (alternative to positional)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir_flag",
        help="Output dir (alternative to positional)",
    )
    parser.add_argument("--no-emoji", action="store_true", help="Disable emoji")
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()
    configure_stdio(args.no_emoji)
    opt_file = args.optimization_file or args.input_file
    out_dir_arg = args.output_dir or args.output_dir_flag
    if not opt_file:
        parser.error("No optimization input file provided (positional or -i/--input)")
    df = pd.read_csv(opt_file)
    # Minimal behavior: pick top 2 combos by quality_score
    combos = []
    # Coerce numeric columns where possible to avoid aggregation errors.
    # Use a strict conversion attempt and fall back to the original series
    # if any values are non-convertible. This avoids the deprecated
    # `errors='ignore'` behavior which emits a FutureWarning in pandas.
    for col in df.columns:
        if col in ("atlas", "connectivity_metric"):
            continue
        try:
            converted = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            # column contains non-numeric values; leave as-is
            continue
        else:
            df[col] = converted
    # Compute group-wise numeric means only
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    group_keys = []
    if "atlas" in df.columns:
        group_keys.append("atlas")
    if "connectivity_metric" in df.columns:
        group_keys.append("connectivity_metric")

    if group_keys and numeric_cols:
        grouped = df.groupby(group_keys)[numeric_cols].mean().reset_index()
    elif group_keys:
        # No numeric cols: deduplicate
        grouped = df.drop_duplicates(subset=group_keys)
    else:
        # As a last resort, group by connectivity_metric if present, else use whole df
        if "connectivity_metric" in df.columns and numeric_cols:
            grouped = (
                df.groupby("connectivity_metric")[numeric_cols].mean().reset_index()
            )
            group_keys = ["connectivity_metric"]
        elif "connectivity_metric" in df.columns:
            grouped = df.drop_duplicates(subset=["connectivity_metric"]).reset_index(
                drop=True
            )
            group_keys = ["connectivity_metric"]
        else:
            grouped = df.copy()

    # Choose top combos by quality_score where available
    if "quality_score" in grouped.columns:
        try:
            top = grouped.nlargest(2, "quality_score")
        except Exception:
            top = grouped.sort_values(by="quality_score", ascending=False).head(2)
    else:
        top = grouped.head(2)
    for _, r in top.iterrows():
        # r may lack 'atlas' or 'connectivity_metric' depending on upstream CSV shape
        atlas_val = r["atlas"] if "atlas" in r.index else None
        metric_val = (
            r["connectivity_metric"] if "connectivity_metric" in r.index else None
        )
        quality_val = None
        if "quality_score" in r.index:
            try:
                quality_val = float(r["quality_score"])
            except Exception:
                quality_val = None
        # Build combo dict with available keys
        combo = {}
        if atlas_val is not None:
            combo["atlas"] = atlas_val
        else:
            combo["atlas"] = None
        if metric_val is not None:
            combo["connectivity_metric"] = metric_val
        else:
            combo["connectivity_metric"] = None
        combo["quality_score"] = quality_val if quality_val is not None else 0.0
        combos.append(combo)
    out_dir = Path(out_dir_arg) if out_dir_arg else Path("03_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "optimal_combinations.json", "w") as f:
        json.dump(combos, f, indent=2)
    print(f"Prepared {len(combos)} optimal combinations and saved to {out_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
