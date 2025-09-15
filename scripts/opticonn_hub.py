#!/usr/bin/env python3
"""
OptiConn Hub CLI
=================

Central CLI that orchestrates optimization, analysis, and pipeline steps.
This module replaces the root-level braingraph.py by living inside scripts/
and resolving paths relative to the repository root automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return repository root directory (parent of scripts/)."""
    # This file lives at <repo>/scripts/opticonn_hub.py
    return Path(__file__).resolve().parent.parent


def _abs(path_like: str | os.PathLike | None) -> str | None:
    if not path_like:
        return None
    return str(Path(path_like).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OptiConn - Unbiased, modality-agnostic connectomics optimization & analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  opticonn optimize -i /path/to/fz -o studies/run1
  opticonn analyze -i /path/to/fz --optimal-config studies/run1/optimize/optimization_results/top3_candidates.json -o studies/run1
  opticonn pipeline --step all -i /path/to/fz -o studies/run2
        """,
    )

    parser.add_argument('--version', action='version', version='OptiConn v2.0.0')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # optimize
    p_opt = subparsers.add_parser('optimize', help='Find optimal parameters using cross-validation')
    p_opt.add_argument('-i', '--data-dir', required=True)
    p_opt.add_argument('-o', '--output-dir', required=True)
    p_opt.add_argument('--config')
    p_opt.add_argument('--quick', action='store_true')
    p_opt.add_argument('--subjects', type=int, default=3)

    # analyze/apply
    for name in ('analyze', 'apply'):
        p = subparsers.add_parser(name, help='Run full analysis with optimal parameters')
        p.add_argument('-i', '--data-dir', required=True)
        p.add_argument('--optimal-config', required=True)
        p.add_argument('-o', '--output-dir', default='analysis_results')
        p.add_argument('--outlier-detection', action='store_true')
        p.add_argument('--skip-extraction', action='store_true')
        p.add_argument('--interactive', action='store_true')
        p.add_argument('--candidate-index', type=int, default=1)
        p.add_argument('--quiet', action='store_true')

    # pipeline
    p_pipe = subparsers.add_parser('pipeline', help='Advanced pipeline execution (steps 01–03)')
    p_pipe.add_argument('--step', default='all', choices=['01', '02', '03', 'all', 'analysis'])
    p_pipe.add_argument('-i', '--input')
    p_pipe.add_argument('-o', '--output')
    p_pipe.add_argument('--config')
    p_pipe.add_argument('--data-dir')
    p_pipe.add_argument('--cross-validated-config')
    p_pipe.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    root = repo_root()
    scripts_dir = root / 'scripts'

    if args.command == 'optimize':
        cmd = [
            sys.executable, str(scripts_dir / 'cross_validation_bootstrap_optimizer.py'),
            '--data-dir', _abs(args.data_dir),
            '--output-dir', _abs(args.output_dir),
        ]
        if args.config:
            cmd += ['--config', _abs(args.config)]
        elif args.quick:
            cmd += ['--config', str(root / 'configs' / 'quick_validation_config.json')]

        print(f"🚀 Running: {' '.join(cmd)}")
        import subprocess
        try:
            subprocess.run(cmd, check=True)
            print("✅ Parameter optimization completed successfully!")
            print(f"📋 Results saved to: {Path(args.output_dir) / 'optimize'}")
            top3 = Path(args.output_dir) / 'optimize' / 'optimization_results' / 'top3_candidates.json'
            print(f"👉 Next: opticonn analyze -i {args.data_dir} --optimal-config {top3} -o {args.output_dir} --interactive")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Optimization failed with error code {e.returncode}")
            return e.returncode

    if args.command in ('analyze', 'apply'):
        # Determine if optimal-config is list (optimal_combinations.json) or dict
        import json
        cfg_path = Path(args.optimal_config)
        try:
            cfg_json = json.loads(Path(cfg_path).read_text())
        except Exception:
            cfg_json = None

        out_selected = Path(args.output_dir) / 'selected'

        if isinstance(cfg_json, list):
            # Rank choices and pick candidate
            def score(obj):
                for k in ('average_score', 'score', 'pure_qa_score', 'quality_score'):
                    v = obj.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
                pw = obj.get('per_wave')
                if isinstance(pw, list):
                    vals = [w.get('score') for w in pw if isinstance(w, dict) and isinstance(w.get('score'), (int, float))]
                    if vals:
                        return float(sum(vals) / len(vals))
                return 0.0

            ranked = sorted(cfg_json, key=score, reverse=True)
            idx = max(1, min(args.candidate_index, len(ranked))) - 1
            chosen = ranked[idx]

            # Resolve DSI Studio command
            dsi_cmd = os.environ.get('DSI_STUDIO_CMD')
            if not dsi_cmd and (root / 'configs' / 'braingraph_default_config.json').exists():
                try:
                    default_cfg = json.loads((root / 'configs' / 'braingraph_default_config.json').read_text())
                    dsi_cmd = default_cfg.get('dsi_studio_cmd')
                except Exception:
                    dsi_cmd = None
            if not dsi_cmd:
                dsi_cmd = '/Applications/dsi_studio.app/Contents/MacOS/dsi_studio' if sys.platform == 'darwin' else 'dsi_studio'

            extraction_cfg = {
                "description": "Extraction from selection (optimal_combinations.json)",
                "atlases": [chosen['atlas']],
                "connectivity_values": [chosen['connectivity_metric']],
                "dsi_studio_cmd": dsi_cmd,
            }
            out_selected.mkdir(parents=True, exist_ok=True)
            extraction_cfg_path = out_selected / 'extraction_from_selection.json'
            extraction_cfg_path.write_text(json.dumps(extraction_cfg, indent=2))

            cmd = [
                sys.executable, str(root / 'run_pipeline.py'),
                '--data-dir', _abs(args.data_dir),
                '--output', str(out_selected),
                '--extraction-config', str(extraction_cfg_path),
                '--step', 'analysis' if args.skip_extraction else 'all',
            ]
            if args.quiet:
                cmd.append('--quiet')
        else:
            # Treat as cross-validated dict config
            cmd = [
                sys.executable, str(root / 'run_pipeline.py'),
                '--cross-validated-config', _abs(args.optimal_config),
                '--data-dir', _abs(args.data_dir),
                '--output', str(out_selected),
                '--step', 'analysis' if args.skip_extraction else 'all',
            ]
            if args.quiet:
                cmd.append('--quiet')

        print(f"🚀 Running: {' '.join(cmd)}")
        import subprocess
        try:
            subprocess.run(cmd, check=True)
            print("✅ Complete analysis finished successfully!")
            print(f"📋 Results available in: {out_selected}")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis failed with error code {e.returncode}")
            return e.returncode

    if args.command == 'pipeline':
        cmd = [sys.executable, str(root / 'run_pipeline.py')]
        if args.step:
            cmd += ['--step', args.step]
        if args.input:
            cmd += ['--input', _abs(args.input)]
        if args.output:
            cmd += ['--output', _abs(args.output)]
        if args.config:
            cmd += ['--config', _abs(args.config)]
        else:
            # Default extraction config
            cmd += ['--extraction-config', str(root / 'configs' / 'braingraph_default_config.json')]
        if args.data_dir:
            cmd += ['--data-dir', _abs(args.data_dir)]
        if args.cross_validated_config:
            cmd += ['--cross-validated-config', _abs(args.cross_validated_config)]
        if args.quiet:
            cmd.append('--quiet')

        print(f"🚀 Running: {' '.join(cmd)}")
        import subprocess
        try:
            subprocess.run(cmd, check=True)
            print("✅ Pipeline execution completed!")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline failed with error code {e.returncode}")
            return e.returncode

    print("Unknown command")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
