#!/usr/bin/env python3
"""
Aggregate Top Candidates Across Waves
====================================

Reads optimal selections from multiple wave outputs and computes a
cross-wave ranking of atlas/metric candidates. Writes the best three
to top3_candidates.json so a user can choose which to run on all subjects.

Inputs per wave:
  <wave_dir>/selected_combinations/optimal_combinations.json  (list)

Outputs in out_dir:
  top3_candidates.json
  all_candidates_ranked.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _load_wave_candidates(wave_dir: Path) -> List[Dict]:
    """Load optimal combinations from a wave directory.

    Supports both current layout and legacy layout:
      - current: <wave_dir>/03_selection/optimal_combinations.json
      - legacy:  <wave_dir>/selected_combinations/optimal_combinations.json
    """
    candidate_paths = [
        wave_dir / "03_selection" / "optimal_combinations.json",
        wave_dir / "selected_combinations" / "optimal_combinations.json",
    ]

    for sel_file in candidate_paths:
        if sel_file.exists():
            try:
                data = json.load(sel_file.open())
                return data if isinstance(data, list) else []
            except Exception:
                return []
    return []


def _load_wave_selected_params(wave_dir: Path) -> Optional[Dict]:
    """Load selected parameter config for a wave, if present.

    Expect file: <wave_dir>/selected_parameters.json with shape {"selected_config": {...}}
    Returns the selected config dict or None.
    """
    sel = wave_dir / "selected_parameters.json"
    if sel.exists():
        try:
            data = json.load(sel.open())
            if isinstance(data, dict):
                cfg = data.get("selected_config") or data
                if isinstance(cfg, dict):
                    return cfg
        except Exception:
            return None
    return None


def aggregate_top_candidates(
    wave_dirs: List[Path], out_dir: Path, top_n: int = 3
) -> Dict:
    # key: (atlas, metric, tract_count) -> list of scores across waves
    scores: Dict[Tuple[str, str, Optional[int]], List[float]] = {}
    details: Dict[Tuple[str, str, Optional[int]], List[Dict]] = {}

    for wave in wave_dirs:
        candidates = _load_wave_candidates(wave)
        params = _load_wave_selected_params(wave) or {}
        # Build a concise parameter snapshot if possible
        param_snapshot = None
        if isinstance(params, dict):
            try:
                tp = params.get("tracking_parameters") or {}
                conn = params.get("connectivity_options") or {}
                param_snapshot = {
                    "tract_count": params.get("tract_count"),
                    "connectivity_threshold": conn.get("connectivity_threshold"),
                    "tracking_parameters": {
                        "fa_threshold": tp.get("fa_threshold"),
                        "turning_angle": tp.get("turning_angle"),
                        "step_size": tp.get("step_size"),
                        "smoothing": tp.get("smoothing"),
                        "min_length": tp.get("min_length"),
                        "max_length": tp.get("max_length"),
                        "track_voxel_ratio": tp.get("track_voxel_ratio"),
                        "dt_threshold": tp.get("dt_threshold"),
                    },
                }
            except Exception:
                param_snapshot = None
        # Try to extract tract_count from selected config
        tract_count = None
        if isinstance(params, dict):
            tract_count = params.get("tract_count")
            # Also check nested sweep_meta.choice if present
            if tract_count is None:
                sweep_meta = params.get("sweep_meta") or {}
                choice = sweep_meta.get("choice") or {}
                tract_count = choice.get("tract_count")
        for c in candidates:
            key = (c.get("atlas"), c.get("connectivity_metric"), tract_count)
            score = c.get("pure_qa_score", c.get("quality_score", 0.0))
            scores.setdefault(key, []).append(float(score))
            details.setdefault(key, []).append(
                {
                    "wave": wave.name,
                    "score": float(score),
                    "qa_penalties": c.get("qa_penalties"),
                    "qa_methodology": c.get("qa_methodology"),
                    "tract_count": tract_count,
                    "parameters": param_snapshot,
                }
            )

    ranked: List[Dict] = []
    for (atlas, metric, tract_count), vals in scores.items():
        avg = sum(vals) / len(vals)
        ranked.append(
            {
                "atlas": atlas,
                "connectivity_metric": metric,
                "tract_count": tract_count,
                "average_score": avg,
                "waves_considered": len(vals),
                "per_wave": details[(atlas, metric, tract_count)],
                # Surface parameters from first available wave snapshot
                "parameters": next(
                    (
                        d.get("parameters")
                        for d in details[(atlas, metric, tract_count)]
                        if d.get("parameters")
                    ),
                    None,
                ),
            }
        )

    ranked.sort(key=lambda x: x["average_score"], reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_file = out_dir / "all_candidates_ranked.json"
    top3_file = out_dir / "top3_candidates.json"

    json.dump(ranked, all_file.open("w"), indent=2)
    json.dump(ranked[:top_n], top3_file.open("w"), indent=2)

    return {
        "all_candidates_ranked": str(all_file),
        "top3_candidates": str(top3_file),
        "count": len(ranked),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate top candidates across waves"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: validate inputs and show summary without writing outputs",
    )
    # Print help when no args provided
    import sys

    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    parser.add_argument("out_dir", help="Output directory for aggregated candidates")
    parser.add_argument("wave_dirs", nargs="+", help="Wave output directories")
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="How many top candidates to write (default: 3)",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN] Aggregate top candidates preview")
        print(f"[DRY-RUN] Wave dirs: {args.wave_dirs}")
        print(f"[DRY-RUN] Out dir (would be): {args.out_dir}")
        print(f"[DRY-RUN] top-n: {args.top_n}")
        return 0

    waves = [Path(w) for w in args.wave_dirs]
    out_dir = Path(args.out_dir)
    res = aggregate_top_candidates(waves, out_dir, args.top_n)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
