#!/usr/bin/env python3
"""
Braingraph - Advanced Connectomics Analysis Pipeline
==================================================

A comprehensive tool for brain connectivity analysis with three main use cases:

1. Parameter Optimization:
   braingraph optimize --data-dir /path/to/data --output-dir results/
   
2. Full Analysis with Optimal Parameters:
   braingraph analyze --data-dir /path/to/data --optimal-config optimal_params.json
   
3. Flexible Pipeline Execution:
   braingraph pipeline --step all --input matrices/ --output results/

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
    braingraph optimize --data-dir /path/to/fib_files --output-dir optimization_results
    
    # 2. Analyze all subjects with optimal parameters
    braingraph analyze --data-dir /path/to/fib_files --optimal-config optimal_params.json --outlier-detection
    
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
    optimize_parser.add_argument('--data-dir', required=True, help='Directory with .fib.gz/.fz files')
    optimize_parser.add_argument('--output-dir', required=True, help='Output directory for optimization results')
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
    analyze_parser.add_argument('--data-dir', required=True, help='Directory with .fib.gz/.fz files')
    analyze_parser.add_argument('--optimal-config', required=True, help='Optimal parameters configuration (from optimize command)')
    analyze_parser.add_argument('--output-dir', default='analysis_results', help='Output directory (default: analysis_results)')
    analyze_parser.add_argument('--outlier-detection', action='store_true', help='Enable outlier detection in final analysis')
    analyze_parser.add_argument('--skip-extraction', action='store_true', help='Skip extraction if matrices already exist')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # 3. PIPELINE command - Flexible pipeline (advanced)
    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Advanced pipeline execution with custom configurations',
        description='Flexible pipeline runner for advanced users with custom configurations.'
    )
    pipeline_parser.add_argument('--step', default='all', 
                               choices=['01', '02', '03', '04', 'all', 'analysis'],
                               help='Pipeline step to run')
    pipeline_parser.add_argument('--input', help='Input directory')
    pipeline_parser.add_argument('--output', help='Output directory')
    pipeline_parser.add_argument('--config', help='Custom configuration file')
    pipeline_parser.add_argument('--data-dir', help='Data directory for step 01')
    pipeline_parser.add_argument('--cross-validated-config', help='Cross-validated configuration file')
    pipeline_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
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
    else:
        print(f"Unknown command: {args.command}")
        return 1


def execute_optimize(args):
    """Execute parameter optimization workflow."""
    print(f"🔬 PARAMETER OPTIMIZATION")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    
    # Construct command for cross_validation_bootstrap_optimizer.py
    cmd = [
        sys.executable, 'cross_validation_bootstrap_optimizer.py',
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
        print(f"📋 Results saved to: {args.output_dir}")
        print(f"🎯 Use the optimal configuration file with 'braingraph analyze' command")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Optimization failed with error code {e.returncode}")
        return e.returncode


def execute_analyze(args):
    """Execute complete analysis with optimal parameters."""
    print(f"🧠 COMPLETE CONNECTOMICS ANALYSIS")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"⚙️  Optimal config: {args.optimal_config}")
    print(f"📊 Output directory: {args.output_dir}")
    
    # Construct command for run_pipeline.py with cross-validated config
    cmd = [
        sys.executable, 'run_pipeline.py',
        '--cross-validated-config', args.optimal_config,
        '--data-dir', args.data_dir,
        '--step', 'analysis' if args.skip_extraction else 'all'
    ]
    
    if args.verbose:
        cmd.append('--verbose')
    
    if args.outlier_detection:
        print(f"🔍 Outlier detection: ENABLED")
        # Add outlier detection flag if available in run_pipeline.py
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Complete analysis finished successfully!")
        print(f"📋 Results available in: {args.output_dir}")
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
    cmd = [sys.executable, 'run_pipeline.py']
    
    if args.step:
        cmd.extend(['--step', args.step])
    if args.input:
        cmd.extend(['--input', args.input])
    if args.output:
        cmd.extend(['--output', args.output])
    if args.config:
        cmd.extend(['--config', args.config])
    if args.data_dir:
        cmd.extend(['--data-dir', args.data_dir])
    if args.cross_validated_config:
        cmd.extend(['--cross-validated-config', args.cross_validated_config])
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


if __name__ == '__main__':
    sys.exit(main())
