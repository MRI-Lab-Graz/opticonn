# Vendored metric_optimizer (light adaptation)

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
import re
from pathlib import Path

from scripts.utils.runtime import configure_stdio

# Minimal reimplementation: wrapper around the original MetricOptimizer class behavior
# For brevity this file exposes a simple CLI compatible interface used by optimizer flow


class MetricOptimizer:
    def __init__(self, config=None):
        self.config = config or {}
        self.weight_factors = self.config.get("weight_factors", {})
        self.quality_threshold = self.config.get("quality_threshold", 0.65)

    def optimize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def _safe_numeric(series: pd.Series) -> pd.Series:
            if series is None:
                return pd.Series([], dtype=float)
            values = pd.to_numeric(series, errors="coerce")
            return values

        def _normalize(series: pd.Series) -> pd.Series:
            series = _safe_numeric(series)
            if series.empty:
                return pd.Series([0.5] * len(df), index=df.index, dtype=float)
            if series.isna().all():
                return pd.Series([0.5] * len(series), index=series.index, dtype=float)
            filled = series.fillna(series.mean())
            min_v = filled.min()
            max_v = filled.max()
            if max_v > min_v:
                return (filled - min_v) / (max_v - min_v)
            return pd.Series([0.5] * len(filled), index=filled.index, dtype=float)

        metric_pattern = re.compile(r"\.([^.]+)\.\.pass\.network_measures$")

        def _extract_metric(identifier: str) -> str:
            if not isinstance(identifier, str):
                return "unknown"
            match = metric_pattern.search(identifier)
            if match:
                return match.group(1)
            primary = identifier.split("..")[0]
            parts = primary.split(".")
            if parts:
                candidate = parts[-1]
                if candidate:
                    return candidate
            return "unknown"

        def _extract_atlas(identifier: str) -> str:
            if not isinstance(identifier, str):
                return "unknown"
            after = identifier.split(".tt.gz.")
            if len(after) >= 2:
                atlas_part = after[1].split(".")[0]
                if atlas_part:
                    return atlas_part
            return "unknown"

        if "connectivity_metric" not in df.columns:
            df["connectivity_metric"] = df["subject_id"].apply(_extract_metric)
        else:
            needs_fix = df["connectivity_metric"].isna() | (
                df["connectivity_metric"].astype(str).str.lower() == "unknown"
            )
            if needs_fix.any():
                df.loc[needs_fix, "connectivity_metric"] = df.loc[
                    needs_fix, "subject_id"
                ].apply(_extract_metric)

        if "atlas" not in df.columns:
            df["atlas"] = df["subject_id"].apply(_extract_atlas)

        density_norm = _normalize(df.get("density"))
        eff_norm = (
            _normalize(df.get("global_efficiency(weighted)"))
            if "global_efficiency(weighted)" in df.columns
            else pd.Series([0.5] * len(df), index=df.index, dtype=float)
        )

        composite = 0.7 * density_norm + 0.3 * eff_norm
        composite = composite.clip(0.0, 1.0)
        df["quality_score"] = composite
        df["quality_score_raw"] = composite
        df["meets_quality_threshold"] = df["quality_score"] >= self.quality_threshold
        df["recommended"] = False

        for metric_name in df["connectivity_metric"].unique():
            subset = df[df["connectivity_metric"] == metric_name]
            if subset.empty:
                continue
            best_idx = subset["quality_score"].idxmax()
            df.loc[best_idx, "recommended"] = True

        return df


def main():
    import argparse, sys

    parser = argparse.ArgumentParser(description="Metric Optimizer (vendored minimal)")
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    parser.add_argument("input_file", nargs="?", help="CSV input")
    parser.add_argument("output_dir", nargs="?", help="Output dir")
    parser.add_argument("--no-emoji", action="store_true", default=None)
    args = parser.parse_args()
    configure_stdio(args.no_emoji)
    if not args.input_file or not args.output_dir:
        parser.print_help()
        return 2
    df = pd.read_csv(args.input_file)
    mo = MetricOptimizer()
    out = mo.optimize_metrics(df)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "optimized_metrics.csv", index=False)
    (out_dir / "optimization_report.txt").write_text(
        "Vendored metric optimizer report\n"
    )
    print("Saved optimized_metrics.csv")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
