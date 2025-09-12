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
from typing import Dict, List, Tuple


def _load_wave_candidates(wave_dir: Path) -> List[Dict]:
    sel_file = wave_dir / 'selected_combinations' / 'optimal_combinations.json'
    if not sel_file.exists():
        return []
    try:
        data = json.load(sel_file.open())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def aggregate_top_candidates(wave_dirs: List[Path], out_dir: Path, top_n: int = 3) -> Dict:
    # key: (atlas, metric) -> list of scores across waves
    scores: Dict[Tuple[str, str], List[float]] = {}
    details: Dict[Tuple[str, str], List[Dict]] = {}

    for wave in wave_dirs:
        candidates = _load_wave_candidates(wave)
        for c in candidates:
            key = (c.get('atlas'), c.get('connectivity_metric'))
            score = c.get('pure_qa_score', c.get('quality_score', 0.0))
            scores.setdefault(key, []).append(float(score))
            details.setdefault(key, []).append({
                'wave': wave.name,
                'score': float(score),
                'qa_penalties': c.get('qa_penalties'),
                'qa_methodology': c.get('qa_methodology'),
            })

    ranked: List[Dict] = []
    for (atlas, metric), vals in scores.items():
        avg = sum(vals) / len(vals)
        ranked.append({
            'atlas': atlas,
            'connectivity_metric': metric,
            'average_score': avg,
            'waves_considered': len(vals),
            'per_wave': details[(atlas, metric)],
        })

    ranked.sort(key=lambda x: x['average_score'], reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_file = out_dir / 'all_candidates_ranked.json'
    top3_file = out_dir / 'top3_candidates.json'

    json.dump(ranked, all_file.open('w'), indent=2)
    json.dump(ranked[:top_n], top3_file.open('w'), indent=2)

    return {
        'all_candidates_ranked': str(all_file),
        'top3_candidates': str(top3_file),
        'count': len(ranked),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate top candidates across waves')
    parser.add_argument('out_dir', help='Output directory for aggregated candidates')
    parser.add_argument('wave_dirs', nargs='+', help='Wave output directories')
    parser.add_argument('--top-n', type=int, default=3, help='How many top candidates to write (default: 3)')
    args = parser.parse_args()

    waves = [Path(w) for w in args.wave_dirs]
    out_dir = Path(args.out_dir)
    res = aggregate_top_candidates(waves, out_dir, args.top_n)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
