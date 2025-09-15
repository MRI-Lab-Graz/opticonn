#!/usr/bin/env python3
"""
Braingraph - Advanced Connectomics Analysis Pipeline
==================================================

⚠️  BEFORE TESTING: Ensure environment is activated!
   source braingraph_pipeline/bin/activate

A comprehensive tool for brain connectivity analysis with three main use cases:

1. Parameter Optimization:
   braingraph optimize -i /path/to/data -o results/
   
2. Full Analysis with Optimal Parameters:
   braingraph analyze -i /path/to/data --optimal-config optimal_params.json
   
3. Flexible Pipeline Execution:
   braingraph pipeline --step all -i matrices/ -o results/

Features:
- Cross-validated parameter optimization
- Automated outlier detection
- Multiple connectivity metrics (FA, QA, count, ncount2)
- Multiple brain atlases support
- Bootstrap quality assurance
- Statistical analysis and visualization

Author: Braingraph Pipeline Team
"""

import argparse
import sys
import os
from pathlib import Path


def main():
    """Main entry point with subcommand structure."""
    parser = argparse.ArgumentParser(
        description="Braingraph - Advanced Connectomics Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 1. Find optimal parameters (cross-validation)
    braingraph optimize -i /path/to/fib_files -o optimization_results
    
    # 2. Analyze all subjects with optimal parameters
    braingraph analyze -i /path/to/fib_files --optimal-config optimal_params.json --outlier-detection
    
    # 3. Run flexible pipeline (advanced users)
    braingraph pipeline --step all --data-dir /path/to/data --config custom_config.json

Use Cases:
    optimize    Cross-validated parameter optimization for best connectivity extraction
    analyze     Complete connectomics analysis with validated parameters + outlier detection  
    pipeline    Flexible pipeline execution with custom configurations (advanced)
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version='Braingraph v2.0.0')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 1. OPTIMIZE command - Parameter optimization
    optimize_parser = subparsers.add_parser(
        'optimize', 
        help='Find optimal parameters using cross-validation',
        description='Run cross-validated parameter optimization to find the best DSI Studio parameters for your dataset.'
    )
    optimize_parser.add_argument('-i', '--data-dir', required=True, help='Directory with .fib.gz/.fz files')
    optimize_parser.add_argument('-o', '--output-dir', required=True, help='Output directory for optimization results')
    optimize_parser.add_argument('--config', help='Optimization configuration file (optional)')
    optimize_parser.add_argument('--quick', action='store_true', help='Quick validation (4 parameter combinations)')
    optimize_parser.add_argument('--subjects', type=int, default=3, help='Number of subjects per validation wave (default: 3)')
    optimize_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # 2. ANALYZE command - Full analysis with optimal parameters
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Complete connectomics analysis with optimal parameters',
        description='Run complete connectomics analysis on all subjects using validated optimal parameters.'
    )
    analyze_parser.add_argument('-i', '--data-dir', required=True, help='Directory with .fib.gz/.fz files')
    analyze_parser.add_argument('--optimal-config', required=True, help='Optimal parameters configuration (from optimize command)')
    analyze_parser.add_argument('-o', '--output-dir', default='analysis_results', help='Output directory (default: analysis_results)')
    analyze_parser.add_argument('--outlier-detection', action='store_true', help='Enable outlier detection in final analysis')
    analyze_parser.add_argument('--skip-extraction', action='store_true', help='Skip extraction if matrices already exist')
    analyze_parser.add_argument('--include-stats', action='store_true', help='Include Step 04 (statistical analysis)')
    analyze_parser.add_argument('--interactive', action='store_true', help='Interactively choose candidate when optimal-config contains a list')
    analyze_parser.add_argument('--candidate-index', type=int, default=1, help='When optimal-config is a list (optimal_combinations.json), choose candidate 1..N (default: 1)')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    analyze_parser.add_argument('--quiet', action='store_true', help='Reduce console output from pipeline')

    # 2b. APPLY command - Alias for analyze (UX)
    apply_parser = subparsers.add_parser(
        'apply',
        help='Alias for analyze: apply optimal parameters to all subjects',
        description='Alias for analyze. Runs complete connectomics analysis with validated optimal parameters.'
    )
    apply_parser.add_argument('-i', '--data-dir', required=True, help='Directory with .fib.gz/.fz files')
    apply_parser.add_argument('--optimal-config', required=True, help='Optimal parameters configuration (from optimize command)')
    apply_parser.add_argument('-o', '--output-dir', default='analysis_results', help='Output directory (default: analysis_results)')
    apply_parser.add_argument('--outlier-detection', action='store_true', help='Enable outlier detection in final analysis')
    apply_parser.add_argument('--skip-extraction', action='store_true', help='Skip extraction if matrices already exist')
    apply_parser.add_argument('--include-stats', action='store_true', help='Include Step 04 (statistical analysis)')
    apply_parser.add_argument('--interactive', action='store_true', help='Interactively choose candidate when optimal-config contains a list')
    apply_parser.add_argument('--candidate-index', type=int, default=1, help='When optimal-config is a list (optimal_combinations.json), choose candidate 1..N (default: 1)')
    apply_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    apply_parser.add_argument('--quiet', action='store_true', help='Reduce console output from pipeline')
    
    # 3. PIPELINE command - Flexible pipeline (advanced)
    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Advanced pipeline execution with custom configurations',
        description='Flexible pipeline runner for advanced users with custom configurations.'
    )
    pipeline_parser.add_argument('--step', default='all', 
                               choices=['01', '02', '03', '04', 'all', 'analysis'],
                               help='Pipeline step to run')
    pipeline_parser.add_argument('-i', '--input', help='Input directory')
    pipeline_parser.add_argument('-o', '--output', help='Output directory')
    pipeline_parser.add_argument('--config', help='Custom configuration file')
    pipeline_parser.add_argument('--data-dir', help='Data directory for step 01')
    pipeline_parser.add_argument('--cross-validated-config', help='Cross-validated configuration file')
    pipeline_parser.add_argument('--include-stats', action='store_true', help='Include Step 04 when using --step all')
    pipeline_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    pipeline_parser.add_argument('--quiet', action='store_true', help='Reduce console output from pipeline')

    # 4. SWEEP command - Generate/run parameter sweep
    sweep_parser = subparsers.add_parser(
        'sweep',
        help='Generate (and optionally run) parameter sweep',
        description='Create parameter combinations from sweep_parameters (grid/random/lhs) and write derived configs.'
    )
    sweep_parser.add_argument('--config', default='configs/braingraph_default_config.json', help='Base config with sweep_parameters')
    sweep_parser.add_argument('-i', '--data-dir', required=True, help='Directory with .fib.gz/.fz files')
    sweep_parser.add_argument('-o', '--output-dir', default='sweep_runs', help='Output directory for generated configs')
    sweep_parser.add_argument('--quick', action='store_true', help='Use quick_sweep ranges if available')
    sweep_parser.add_argument('--execute', action='store_true', help='Run pipeline for generated configs (sequential)')
    sweep_parser.add_argument('--max-executions', type=int, default=0, help='Max runs to execute immediately (0=all)')
    sweep_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    sweep_parser.add_argument('--quiet', action='store_true', help='Reduce console output during run')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Import and execute the appropriate command
    if args.command == 'optimize':
        return execute_optimize(args)
    elif args.command == 'analyze':
        return execute_analyze(args)
    elif args.command == 'pipeline':
        return execute_pipeline(args)
    elif args.command == 'sweep':
        return execute_sweep(args)
    elif args.command == 'apply':
        return execute_analyze(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def execute_optimize(args):
    """Execute parameter optimization workflow."""
    print(f"🔬 PARAMETER OPTIMIZATION")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    
    # Construct command for cross_validation_bootstrap_optimizer.py
    # Use 'python' to respect the current environment instead of sys.executable
    cmd = [
        'python', 'scripts/cross_validation_bootstrap_optimizer.py',
        '--data-dir', args.data_dir,
        '--output-dir', args.output_dir
    ]
    
    if args.config:
        cmd.extend(['--config', args.config])
    elif args.quick:
        cmd.extend(['--config', 'configs/quick_validation_config.json'])
    
    # Note: cross_validation_bootstrap_optimizer.py doesn't support --verbose yet
    # if args.verbose:
    #     cmd.append('--verbose')
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Parameter optimization completed successfully!")
        print(f"📋 Results saved to: {Path(args.output_dir) / 'optimize'}")
        # Friendly next steps
    top3 = Path(args.output_dir) / 'optimize' / 'optimization_results' / 'top3_candidates.json'
        print(f"📄 If available, Top-3 candidates: {top3}")
    # Recommend using the same root -o so analyze will save under <root>/selected
    print(f"👉 Next: braingraph analyze -i {args.data_dir} --optimal-config {top3} -o {args.output_dir} --interactive")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Optimization failed with error code {e.returncode}")
        return e.returncode


def execute_analyze(args):
    """Execute complete analysis with optimal parameters.

    Accepts either a dict-based cross-validated config or a list-based
    optimal_combinations.json. For lists, writes top3_candidates.json and
    lets the user pick via --candidate-index.
    """
    print(f"🧠 COMPLETE CONNECTOMICS ANALYSIS")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"⚙️  Optimal config: {args.optimal_config}")
    print(f"📊 Output directory: {args.output_dir}")

    # Try to parse optimal-config
    import json
    from pathlib import Path
    cfg_path = Path(args.optimal_config)
    try:
        with cfg_path.open() as f:
            cfg_json = json.load(f)
    except Exception:
        cfg_json = None

    # If it's a list of combinations: build extraction config from choice
    if isinstance(cfg_json, list):
        if not cfg_json:
            print("❌ No candidates in optimal_combinations.json")
            return 2

        def score(obj):
            # Prefer consolidated scores when available
            for k in ('average_score', 'score', 'pure_qa_score', 'quality_score'):
                v = obj.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            # Fallback: average per-wave scores if present
            pw = obj.get('per_wave')
            if isinstance(pw, list):
                vals = [w.get('score') for w in pw if isinstance(w, dict) and isinstance(w.get('score'), (int, float))]
                if vals:
                    return float(sum(vals) / len(vals))
            return 0.0

        ranked = sorted(cfg_json, key=score, reverse=True)
        top3 = ranked[:3]

        # Save top-3 for user transparency
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        top3_file = out_dir / 'top3_candidates.json'
        top3_file.write_text(json.dumps(top3, indent=2))
        print(f"📄 Top-3 candidates saved: {top3_file}")

        if args.interactive and sys.stdin and sys.stdin.isatty():
            print("\nPlease choose a candidate:")
            for i, c in enumerate(ranked[:10], 1):
                sc = score(c)
                print(f"  {i}. {c['atlas']} + {c['connectivity_metric']} (score={sc:.3f})")
            try:
                raw = input(f"Selection [1-{min(10, len(ranked))}] (default {args.candidate_index}): ").strip()
                if raw:
                    args.candidate_index = int(raw)
            except Exception:
                pass
        idx = max(1, min(args.candidate_index, len(ranked))) - 1
        chosen = ranked[idx]
        print(f"🎯 Selected candidate #{idx+1}: {chosen['atlas']} + {chosen['connectivity_metric']} (score={score(chosen):.3f})")

        # Resolve DSI Studio command
        dsi_cmd = os.environ.get('DSI_STUDIO_CMD')
        if not dsi_cmd:
            # Try default config
            try:
                default_cfg_path = Path('configs/braingraph_default_config.json')
                if default_cfg_path.exists():
                    with default_cfg_path.open() as df:
                        default_cfg = json.load(df)
                        dsi_cmd = default_cfg.get('dsi_studio_cmd')
            except Exception:
                dsi_cmd = None
        if not dsi_cmd:
            # macOS default app bundle path as sensible fallback
            if sys.platform == 'darwin':
                dsi_cmd = '/Applications/dsi_studio.app/Contents/MacOS/dsi_studio'
            else:
                dsi_cmd = 'dsi_studio'

        # Build minimal extraction config with resolved DSI Studio path
        extraction_cfg = {
            "description": "Extraction from selection (optimal_combinations.json)",
            "atlases": [chosen['atlas']],
            "connectivity_values": [chosen['connectivity_metric']],
            "dsi_studio_cmd": dsi_cmd
        }
        extraction_cfg_path = out_dir / 'extraction_from_selection.json'
        extraction_cfg_path.write_text(json.dumps(extraction_cfg, indent=2))
        print(f"🧩 Created extraction config: {extraction_cfg_path}")

    # Run pipeline
        cmd = [
            'python', 'run_pipeline.py',
            '--data-dir', args.data_dir,
            '--output', str(Path(args.output_dir) / 'selected'),
            '--extraction-config', str(extraction_cfg_path),
            '--step', 'analysis' if args.skip_extraction else 'all'
        ]
        if args.include_stats and not args.skip_extraction:
            cmd.append('--include-stats')
        if args.verbose:
            cmd.append('--verbose')
        if args.quiet:
            cmd.append('--quiet')
        print(f"🚀 Running: {' '.join(cmd)}")

        import subprocess
        try:
            subprocess.run(cmd, check=True)
            print("✅ Complete analysis finished successfully!")
            print(f"📋 Results available in: {Path(args.output_dir) / 'selected'}")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis failed with error code {e.returncode}")
            return e.returncode

    # Otherwise, treat as cross-validated dict config
    cmd = [
        'python', 'run_pipeline.py',
        '--cross-validated-config', args.optimal_config,
        '--data-dir', args.data_dir,
        '--output', str(Path(args.output_dir) / 'selected'),
        '--step', 'analysis' if args.skip_extraction else 'all'
    ]
    if args.include_stats and not args.skip_extraction:
        cmd.append('--include-stats')
    if args.verbose:
        cmd.append('--verbose')
    if args.quiet:
        cmd.append('--quiet')

    if args.outlier_detection:
        print(f"🔍 Outlier detection: ENABLED")

    print(f"🚀 Running: {' '.join(cmd)}")

    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Complete analysis finished successfully!")
        print(f"📋 Results available in: {Path(args.output_dir) / 'selected'}")
        if args.outlier_detection:
            print(f"📊 Check statistical analysis results for outlier detection")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed with error code {e.returncode}")
        return e.returncode


def execute_pipeline(args):
    """Execute flexible pipeline workflow."""
    print(f"⚙️  FLEXIBLE PIPELINE EXECUTION")
    print(f"📋 Step: {args.step}")
    
    # Construct command for run_pipeline.py with provided arguments
    cmd = ['python', 'run_pipeline.py']
    
    if args.step:
        cmd.extend(['--step', args.step])
    if args.input:
        cmd.extend(['--input', args.input])
    if args.output:
        cmd.extend(['--output', args.output])
    if args.config:
        cmd.extend(['--config', args.config])
    else:
        # Default to braingraph_default_config.json if no config specified
        cmd.extend(['--extraction-config', 'configs/braingraph_default_config.json'])
    if args.data_dir:
        cmd.extend(['--data-dir', args.data_dir])
    if args.cross_validated_config:
        cmd.extend(['--cross-validated-config', args.cross_validated_config])
    if args.include_stats and args.step == 'all':
        cmd.append('--include-stats')
    if args.verbose:
        cmd.append('--verbose')
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Pipeline execution completed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed with error code {e.returncode}")
        return e.returncode


def execute_sweep(args):
    """Generate parameter sweep using run_parameter_sweep.py."""
    print("🧪 PARAMETER SWEEP")
    print(f"🧩 Base config: {args.config}")
    print(f"📁 Output dir: {args.output_dir}")
    cmd = [
        'python', 'scripts/run_parameter_sweep.py',
        '--config', args.config,
        '-o', args.output_dir
    ]
    # Always pass data-dir once
    if args.data_dir:
        cmd.extend(['-i', args.data_dir])
    if args.quick:
        cmd.append('--quick')
    if args.execute:
        cmd.append('--execute')
    if args.max_executions:
        cmd.extend(['--max-executions', str(args.max_executions)])
    if args.verbose:
        cmd.append('--verbose')
    if args.quiet:
        cmd.append('--quiet')
    print(f"🚀 Running: {' '.join(cmd)}")
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("✅ Sweep generation completed")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Sweep failed with error code {e.returncode}")
        return e.returncode


if __name__ == '__main__':
    sys.exit(main())
