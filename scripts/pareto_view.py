#!/usr/bin/env python3
"""
Pareto View Utility
===================

Build a simple multi-objective Pareto view over sweep combinations using
persisted diagnostics. Objectives:
  - Maximize score (quality_score_raw_mean by default; fallback to selection_score)
  - Minimize density corridor deviation (distance to [lo, hi], 0 if inside)
  - Minimize cost (tract_count)

Inputs: one or more wave directories (containing combo_diagnostics.csv)
        or direct paths to combo_diagnostics.csv files.
Outputs: pareto_front.csv (+ optional pareto_front.png if --plot)

Example:
  python scripts/pareto_view.py \
    studies/demo/optimize/bootstrap_qa_wave_1 \
    studies/demo/optimize/bootstrap_qa_wave_2 \
    -o studies/demo/optimize/optimization_results --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


def _load_wave_table(path: Path) -> pd.DataFrame | None:
    """Load combo diagnostics table from a wave directory or CSV file."""
    if path.is_file() and path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(path)
            # Ensure atlas and connectivity_metric columns exist (fill with None if missing)
            for col in ['atlas', 'connectivity_metric']:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return None
    # Try wave dir
    csv_path = path / 'combo_diagnostics.csv'
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None
    # Fallback: build from per-combo diagnostics.json files
    combos_dir = path / 'combos'
    rows: List[Dict] = []
    if combos_dir.exists():
        for child in combos_dir.iterdir():
            if child.is_dir() and child.name.startswith('sweep_'):
                j = child / 'diagnostics.json'
                if not j.exists():
                    continue
                try:
                    rec = json.loads(j.read_text())
                except Exception:
                    continue
                agg = rec.get('aggregates') or {}
                rows.append({
                    'wave': rec.get('wave'),
                    'sweep_id': child.name,
                    'status': rec.get('status'),
                    'combo_index': rec.get('combo_index'),
                    'total_combinations': rec.get('total_combinations'),
                    'sampler': rec.get('sampler'),
                    'thread_count': rec.get('thread_count'),
                    'tract_count': rec.get('tract_count'),
                    'selection_score': rec.get('selection_score'),
                    'quality_score_raw_mean': rec.get('quality_score_raw_mean'),
                    'quality_score_norm_max': rec.get('quality_score_norm_max'),
                    'density_mean': agg.get('density_mean'),
                    'global_efficiency_weighted_mean': agg.get('global_efficiency_weighted_mean'),
                    'small_worldness_binary_mean': agg.get('small_worldness_binary_mean'),
                    'small_worldness_weighted_mean': agg.get('small_worldness_weighted_mean'),
                    'atlas': rec.get('atlas'),
                    'connectivity_metric': rec.get('connectivity_metric'),
                })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _density_deviation(d: float | None, lo: float, hi: float) -> float:
    if d is None or np.isnan(d):
        return float('inf')
    if d < lo:
        return lo - d
    if d > hi:
        return d - hi
    return 0.0


def pareto_front(df: pd.DataFrame, score_col: str, lo: float, hi: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pareto-efficient subset.

    Returns (front_df, all_with_objectives_df)
    """
    data = df.copy()
    # Compute objectives
    data['score'] = data[score_col]
    data['cost'] = data['tract_count']
    data['density_dev'] = [
        _density_deviation(x, lo, hi) for x in data.get('density_mean', pd.Series([np.nan]*len(data)))
    ]
    # Prepare vectors (minimize cost, minimize density_dev, maximize score -> minimize -score)
    objs = data[['cost', 'density_dev', 'score']].to_numpy()
    # Replace NaNs/infs for dominance logic
    objs = np.where(np.isfinite(objs), objs, np.array([np.inf, np.inf, -np.inf]))
    objs[:, 2] = -objs[:, 2]  # negate score for minimization

    n = len(data)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Any j that dominates i?
        dominates = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            is_efficient[i] = False
    front = data[is_efficient].copy()
    return front, data


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate a Pareto view from sweep diagnostics')
    ap.add_argument('--dry-run', action='store_true', default=False,
                    help='Perform a dry-run: summarize inputs and planned outputs without writing files')
    # Print help when no args provided
    import sys
    if len(sys.argv) == 1:
        ap.print_help()
        return 0
    ap.add_argument('inputs', nargs='+', help='Wave directories or combo_diagnostics.csv files')
    ap.add_argument('-o', '--output-dir', required=True, help='Output directory for pareto_front.csv (and plot)')
    ap.add_argument('--score', default='quality_score_raw_mean', choices=['quality_score_raw_mean', 'selection_score'],
                    help='Score column to maximize (default: quality_score_raw_mean)')
    ap.add_argument('--density-range', nargs=2, type=float, default=[0.05, 0.40], metavar=('LO', 'HI'),
                    help='Preferred density corridor (default: 0.05 0.40)')
    ap.add_argument('--plot', action='store_true', help='Also generate a scatter plot (PNG)')
    args = ap.parse_args()
    if args.dry_run:
        print('[DRY-RUN] Pareto view preview')
        print(f"[DRY-RUN] Inputs: {args.inputs}")
        print(f"[DRY-RUN] Output dir: {args.output_dir}")
        print(f"[DRY-RUN] Score: {args.score}")
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for p in args.inputs:
        df = _load_wave_table(Path(p))
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit('No diagnostics found in inputs')
    df_all = pd.concat(frames, ignore_index=True)
    # Keep ok status rows when available
    if 'status' in df_all.columns:
        df_all = df_all[df_all['status'].fillna('ok') == 'ok']

    front, with_obj = pareto_front(df_all, args.score, args.density_range[0], args.density_range[1])
    # Save CSVs
    with_obj_out = out_dir / 'pareto_candidates_with_objectives.csv'
    front_out = out_dir / 'pareto_front.csv'
    with_obj.to_csv(with_obj_out, index=False)
    front.to_csv(front_out, index=False)
    print(f"Saved: {front_out}")
    print(f"Saved: {with_obj_out}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(7.5, 5.5))
            x = with_obj['tract_count']
            y = with_obj['score']
            c = with_obj['density_dev']
            sc = plt.scatter(x, y, c=c, cmap='viridis', s=38, alpha=0.75, edgecolors='none')
            plt.colorbar(sc, label='Density deviation (to corridor)')
            # Overlay Pareto points
            plt.scatter(front['tract_count'], front['score'], marker='*', s=140, color='crimson', label='Pareto front')
            plt.xlabel('Tract count (cost)')
            plt.ylabel(f"Score ({args.score})")
            plt.title('Pareto view: score vs. cost, colored by density deviation')
            plt.legend(loc='best')
            plt.tight_layout()
            fig_path = out_dir / 'pareto_front.png'
            plt.savefig(fig_path, dpi=160)
            print(f"Saved: {fig_path}")
        except Exception as e:
            print(f"Plotting failed: {e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
