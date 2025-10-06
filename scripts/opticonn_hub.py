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

from scripts.utils.runtime import configure_stdio, propagate_no_emoji


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
    parser.add_argument('--no-emoji', action='store_true', help='Disable emoji in console output (useful on limited terminals)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Perform a dry-run: print the command(s) that would be executed without running them')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # review
    p_review = subparsers.add_parser('review', help='Review sweep results and select candidates interactively')
    p_review.add_argument('-o', '--output-dir', required=True, help='Sweep output directory to review')
    p_review.add_argument('--port', type=int, default=8050, help='Port for Dash app')

    # sweep
    p_sweep = subparsers.add_parser('sweep', help='Run parameter sweep using cross-validation')
    p_sweep.add_argument('-i', '--data-dir', required=True)
    p_sweep.add_argument('-o', '--output-dir', required=True)
    p_sweep.add_argument('--config', help='Optional master sweep config (rare). If you want to provide an extraction/sweep config, prefer --extraction-config')
    p_sweep.add_argument('--quick', action='store_true', help='Run a tiny demonstration sweep (uses configs/sweep_micro.json)')
    p_sweep.add_argument('--subjects', type=int, default=3)
    # Advanced/parallel tuning
    p_sweep.add_argument('--max-parallel', type=int, help='Max combinations to run in parallel per wave')
    p_sweep.add_argument('--prune-nonbest', action='store_true', help='Prune non-best combos to save disk space')
    p_sweep.add_argument('--extraction-config', help='Override extraction config for auto-generated waves')
    # Reports are now standard unless --no-report is used
    p_sweep.add_argument('--no-report', action='store_true', help='Skip quick quality and Pareto reports after sweep')
    p_sweep.add_argument('--no-emoji', action='store_true', help='Disable emoji in console output (Windows-safe)')
    p_sweep.add_argument('--no-validation', action='store_true', help='Skip full setup validation before running sweep')
    p_sweep.add_argument('--verbose', action='store_true', help='Show DSI Studio command for each run in main sweep log')

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
        p.add_argument('--no-emoji', action='store_true', help='Disable emoji in console output (Windows-safe)')

    # pipeline
    p_pipe = subparsers.add_parser('pipeline', help='Advanced pipeline execution (steps 01–03)')
    p_pipe.add_argument('--step', default='all', choices=['01', '02', '03', 'all', 'analysis'])
    p_pipe.add_argument('-i', '--input')
    p_pipe.add_argument('-o', '--output')
    p_pipe.add_argument('--config')
    p_pipe.add_argument('--data-dir')
    p_pipe.add_argument('--cross-validated-config')
    p_pipe.add_argument('--quiet', action='store_true')
    p_pipe.add_argument('--no-emoji', action='store_true', help='Disable emoji in console output (Windows-safe)')

    args = parser.parse_args()

    # Print help when called without args
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    root = repo_root()
    scripts_dir = root / 'scripts'
    no_emoji = configure_stdio(args.no_emoji)

    import uuid
    import subprocess
    def validate_json_config(config_path):
        validator_script = str(scripts_dir / "json_validator.py")
        result = subprocess.run([
            sys.executable, validator_script, config_path, "--suggest-fixes"
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("Config validation failed. Exiting.")
            sys.exit(1)

    if args.command == 'review':
        # Launch Dash app for interactive review
        class OptiConnReview:
            def __init__(self, output_dir, port=8050):
                self.output_dir = output_dir
                self.port = port
            def launch(self):
                import subprocess
                dash_app = str(repo_root() / 'scripts' / 'dash_app' / 'app.py')
                cmd = [sys.executable, dash_app, '--output', self.output_dir, '--port', str(self.port)]
                print(f"🚀 Launching OptiConn Review Dashboard: {' '.join(cmd)}")
                subprocess.run(cmd)

        reviewer = OptiConnReview(args.output_dir, args.port)
        reviewer.launch()
        return 0

    if args.command == 'sweep':
        # Run full setup validation unless opted out
        if not getattr(args, 'no_validation', False):
            validate_script = str(scripts_dir / "validate_setup.py")
            # Try to auto-detect config and input for validation
            config_path = args.config or args.extraction_config or str(root / 'configs' / 'braingraph_default_config.json')
            input_path = args.data_dir
            output_path = args.output_dir
            val_args = [sys.executable, validate_script, '--config', config_path, '--output-dir', output_path, '--test-input', input_path]
            result = subprocess.run(val_args, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("❌ Full setup validation failed. Exiting.")
                sys.exit(1)
        # Append UUID to output directory
        unique_id = str(uuid.uuid4())
        sweep_output_dir = f"{_abs(args.output_dir)}/sweep-{unique_id}"
        cmd = [
            sys.executable, str(scripts_dir / 'cross_validation_bootstrap_optimizer.py'),
            '--data-dir', _abs(args.data_dir),
            '--output-dir', sweep_output_dir,
        ]

        # Decide how to interpret provided configuration flags
        chosen_extraction_cfg: str | None = None
        chosen_master_cfg: str | None = None
        if args.quick:
            # Quick demo should use the tiny micro sweep to avoid large grids
            chosen_extraction_cfg = str(root / 'configs' / 'sweep_micro.json')
        if args.extraction_config:
            chosen_extraction_cfg = _abs(args.extraction_config)
        if args.config:
            try:
                import json as _json
                _cfg_path = Path(args.config)
                cfg_txt = _cfg_path.read_text()
                cfg_json = _json.loads(cfg_txt)
                is_master = any(k in cfg_json for k in ('wave1_config', 'wave2_config', 'bootstrap_optimization'))
                is_extraction_like = any(k in cfg_json for k in ('atlases', 'connectivity_values', 'sweep_parameters'))
                if is_master and not is_extraction_like:
                    chosen_master_cfg = _abs(args.config)
                else:
                    chosen_extraction_cfg = _abs(args.config)
            except Exception:
                chosen_extraction_cfg = _abs(args.config)

        # Validate configs before running sweep
        if chosen_master_cfg:
            validate_json_config(chosen_master_cfg)
            cmd += ['--config', chosen_master_cfg]
        if chosen_extraction_cfg:
            validate_json_config(chosen_extraction_cfg)
            cmd += ['--extraction-config', chosen_extraction_cfg]

        if args.subjects:
            cmd += ['--subjects', str(int(args.subjects))]
        if args.max_parallel and int(args.max_parallel) > 1:
            cmd += ['--max-parallel', str(int(args.max_parallel))]
        if args.prune_nonbest:
            cmd += ['--prune-nonbest']
        if args.verbose:
            cmd += ['--verbose']
        if no_emoji:
            cmd.append('--no-emoji')
        if chosen_extraction_cfg:
            print(f"🧪 Using extraction config: {chosen_extraction_cfg}")
        if chosen_master_cfg:
            print(f"📋 Using master optimizer config: {chosen_master_cfg}")
        print(f"🚀 Running: {' '.join(cmd)}")
        print(f"🆔 Sweep output directory: {sweep_output_dir}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Parameter sweep completed successfully!")
            print(f"📋 Results saved to: {sweep_output_dir}/optimize")


            if not getattr(args, 'no_report', False):
                # Autodetect network measures directory for quick quality check
                import glob
                selection_dirs = glob.glob(f"{sweep_output_dir}/optimize/*/03_selection")
                if selection_dirs:
                    matrices_dir = selection_dirs[0]
                    print(f"🔎 Running quick quality check on: {matrices_dir}")
                    qqc_script = str(root / 'scripts' / 'quick_quality_check.py')
                    qqc_args = [sys.executable, qqc_script, matrices_dir]
                    qqc_result = subprocess.run(qqc_args, capture_output=True, text=True)
                    print(qqc_result.stdout)
                    if qqc_result.returncode != 0:
                        print("⚠️  Quick quality check reported issues!")
                else:
                    print("⚠️  Could not find network measures directory for quick quality check.")

                # Always run Pareto report if any wave diagnostics exist
                opt_dir = Path(sweep_output_dir) / 'optimize'
                optimization_results_dir = opt_dir / 'optimization_results'
                optimization_results_dir.mkdir(parents=True, exist_ok=True)
                wave_dirs = []
                for child in opt_dir.iterdir():
                    if child.is_dir() and (child / 'combo_diagnostics.csv').exists():
                        wave_dirs.append(str(child.resolve()))
                if wave_dirs:
                    pareto_cmd = [
                        sys.executable, str(root / 'scripts' / 'pareto_view.py'),
                        *wave_dirs,
                        '-o', str(optimization_results_dir),
                        '--plot'
                    ]
                    print(f"📈 Generating Pareto report: {' '.join(pareto_cmd)}")
                    try:
                        subprocess.run(pareto_cmd, check=True, env=env)
                        print(f"✅ Pareto report written to: {optimization_results_dir}")
                    except subprocess.CalledProcessError as e:
                        print(f"⚠️  Pareto report generation failed with error code {e.returncode}")
                    except Exception as e:
                        print(f"⚠️  Pareto report generation encountered an error: {e}")
                else:
                    print("ℹ️  No wave diagnostics found (combo_diagnostics.csv); skipping Pareto report")

            top3 = Path(sweep_output_dir) / 'optimize' / 'optimization_results' / 'top3_candidates.json'
            print(f"👉 Next: opticonn analyze -i {args.data_dir} --optimal-config {top3} -o {sweep_output_dir} --interactive")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Sweep failed with error code {e.returncode}")
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

            # Tentatively include parameter hints if present on the chosen candidate
            chosen_params = chosen.get('parameters') if isinstance(chosen, dict) else None
            extraction_cfg = {
                "description": "Extraction from selection (optimal_combinations.json)",
                "atlases": [chosen['atlas']],
                "connectivity_values": [chosen['connectivity_metric']],
                "dsi_studio_cmd": dsi_cmd,
            }
            # Merge selected parameter snapshot (non-destructive; downstream config loader tolerates missing fields)
            try:
                if isinstance(chosen_params, dict):
                    if 'tract_count' in chosen_params:
                        extraction_cfg['tract_count'] = chosen_params['tract_count']
                    tp = chosen_params.get('tracking_parameters') or {}
                    if tp:
                        extraction_cfg.setdefault('tracking_parameters', {})
                        for k in ('fa_threshold','turning_angle','step_size','smoothing','min_length','max_length','track_voxel_ratio','dt_threshold'):
                            if tp.get(k) is not None:
                                extraction_cfg['tracking_parameters'][k] = tp.get(k)
                    ct = chosen_params.get('connectivity_threshold')
                    if ct is not None:
                        extraction_cfg.setdefault('connectivity_options', {})
                        extraction_cfg['connectivity_options']['connectivity_threshold'] = ct
            except Exception:
                pass
            out_selected.mkdir(parents=True, exist_ok=True)
            extraction_cfg_path = out_selected / 'extraction_from_selection.json'
            extraction_cfg_path.write_text(json.dumps(extraction_cfg, indent=2))
            # Persist a selected_parameters.json for downstream Step 03 reporting
            try:
                (out_selected / 'selected_parameters.json').write_text(
                    json.dumps({'selected_config': extraction_cfg}, indent=2)
                )
            except Exception:
                pass

            cmd = [
                sys.executable, str(root / 'scripts' / 'run_pipeline.py'),
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
                sys.executable, str(root / 'scripts' / 'run_pipeline.py'),
                '--cross-validated-config', _abs(args.optimal_config),
                '--data-dir', _abs(args.data_dir),
                '--output', str(out_selected),
                '--step', 'analysis' if args.skip_extraction else 'all',
            ]
            if args.quiet:
                cmd.append('--quiet')

        if no_emoji:
            cmd.append('--no-emoji')

        # Validate config before running analysis/apply
        if isinstance(cfg_json, list):
            validate_json_config(str(extraction_cfg_path))
        else:
            validate_json_config(_abs(args.optimal_config))

        print(f"🚀 Running: {' '.join(cmd)}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Complete analysis finished successfully!")
            print(f"📋 Results available in: {out_selected}")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis failed with error code {e.returncode}")
            return e.returncode

    if args.command == 'pipeline':
        cmd = [sys.executable, str(root / 'scripts' / 'run_pipeline.py')]
        config_path = None
        if args.step:
            cmd += ['--step', args.step]
        if args.input:
            cmd += ['--input', _abs(args.input)]
        if args.output:
            cmd += ['--output', _abs(args.output)]
        if args.config:
            config_path = _abs(args.config)
            cmd += ['--extraction-config', config_path]
        else:
            config_path = str(root / 'configs' / 'braingraph_default_config.json')
            cmd += ['--extraction-config', config_path]
        if args.data_dir:
            cmd += ['--data-dir', _abs(args.data_dir)]
        if args.cross_validated_config:
            cmd += ['--cross-validated-config', _abs(args.cross_validated_config)]
        if args.quiet:
            cmd.append('--quiet')
        if no_emoji:
            cmd.append('--no-emoji')

        # Validate config before running pipeline
        if config_path:
            validate_json_config(config_path)

        print(f"🚀 Running: {' '.join(cmd)}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Pipeline execution completed!")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline failed with error code {e.returncode}")
            return e.returncode

    print("Unknown command")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
