#!/usr/bin/env python3
"""
Braingraph Pipeline Runner
=========================

Simplified pipeline runner that chains the core analysis steps.
This replaces the complex bash scripts 02-04 with a clean Python workflow.

Usage:
    python run_pipeline.py [--input organized_matrices_dir] [--output results_dir] [--step STEP]

Steps:
    01: Connectivity extraction (extract_connectivity_matrices.py) with JSON config
    02: Network quality optimization (metric_optimizer.py)
    03: Quality-based selection (optimal_selection.py) 
    04: Statistical analysis (statistical_analysis.py)
    all: Run all steps 01-04 (default)
    analysis: Run only steps 02-04 (skip extraction)

Examples:
    # Full pipeline with JSON config
    python run_pipeline.py --step all --data-dir /path/to/raw/data \\
                          --extraction-config sweep_config.json
    
    # Analysis only (skip extraction)
    python run_pipeline.py --step analysis --input organized_matrices/
    
    # Pilot test
    python run_pipeline.py --step 01 --data-dir /path/to/raw/data \\
                          --extraction-config sweep_config.json --pilot

Author: Braingraph Pipeline Team
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'pipeline_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def run_step(script_name, args, logger, step_name):
    """Run a pipeline step and handle errors."""
    logger.info(f"🚀 Starting {step_name}...")
    logger.info(f"Command: python {script_name} {' '.join(args)}")
    
    try:
        result = subprocess.run(
            ['python', script_name] + args,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✅ {step_name} completed successfully")
        if result.stdout:
            logger.debug(f"Output:\n{result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed with return code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        if e.stdout:
            logger.error(f"Standard output: {e.stdout}")
        return False
    except Exception as e:
        logger.error(f"❌ {step_name} failed with exception: {e}")
        return False

def find_organized_matrices(base_dir):
    """Find the organized matrices directory from step 01."""
    base_path = Path(base_dir)
    
    # Look for organized_matrices directory
    organized_dirs = list(base_path.glob("**/organized_matrices"))
    if organized_dirs:
        return organized_dirs[0]
    
    # Look for directories with atlas structure
    for dir_path in base_path.iterdir():
        if dir_path.is_dir():
            # Check if it has atlas subdirectories
            atlas_dirs = [d for d in dir_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if len(atlas_dirs) > 3:  # Likely contains multiple atlases
                return dir_path
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Braingraph Pipeline Runner - Steps 02-04",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline (steps 02-04)
    python run_pipeline.py --input results/organized_matrices/ --output analysis_results/
    
    # Run only optimization step
    python run_pipeline.py --input organized_matrices/ --step 02
    
    # Run with verbose logging
    python run_pipeline.py --input data/ --output results/ --verbose
    
    # Auto-detect input from step 01 output
    python run_pipeline.py --output analysis_results/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input directory (organized matrices from step 01). Will auto-detect if not specified.'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='analysis_results',
        help='Output directory for analysis results (default: analysis_results)'
    )
    
    parser.add_argument(
        '--step', '-s',
        choices=['01', '02', '03', '04', 'all', 'analysis'],
        default='all',
        help='Pipeline step to run: 01=extraction, 02=optimization, 03=selection, 04=statistics, all=complete pipeline, analysis=steps 02-04 only (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file for pipeline parameters'
    )
    
    # Extraction parameters (Step 01) - JSON Config Approach
    extraction_group = parser.add_argument_group('Step 01: Connectivity Extraction')
    
    extraction_group.add_argument(
        '--data-dir',
        help='Input data directory containing .fib.gz/.fz files (for step 01)'
    )
    
    extraction_group.add_argument(
        '--extraction-config',
        help='JSON configuration file for extraction parameters (like sweep_config.json). '
             'Replaces individual parameter flags. See example configs.'
    )
    
    extraction_group.add_argument(
        '--pilot', action='store_true',
        help='Run pilot test on subset of files'
    )
    
    extraction_group.add_argument(
        '--pilot-count', type=int, default=2,
        help='Number of files for pilot test (default: 2)'
    )
    
    extraction_group.add_argument(
        '--batch', action='store_true',
        help='Batch mode: process all files in data directory'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧠 Braingraph Pipeline Runner")
    logger.info("=" * 50)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Steps to run: {args.step}")
    
    # Find input directory or data directory
    if args.step == '01' or (args.step == 'all' and not args.input):
        # For step 01, we need raw data directory
        if args.data_dir:
            data_path = Path(args.data_dir)
            if not data_path.exists():
                logger.error(f"Data directory does not exist: {data_path}")
                sys.exit(1)
            logger.info(f"📁 Data directory: {data_path}")
        elif args.input:
            data_path = Path(args.input)
            if not data_path.exists():
                logger.error(f"Input directory does not exist: {data_path}")
                sys.exit(1)
            logger.info(f"📁 Data directory: {data_path}")
        else:
            logger.error("For step 01, please specify --data-dir or --input with raw data directory")
            sys.exit(1)
    
    # For steps 02-04, find organized matrices
    if args.step in ['02', '03', '04', 'analysis'] or (args.step == 'all' and args.input):
        if args.input:
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"Input directory does not exist: {input_path}")
                sys.exit(1)
        else:
            logger.info("🔍 Auto-detecting input directory from step 01 output...")
            input_path = find_organized_matrices(Path.cwd())
            if not input_path:
                logger.error("Could not auto-detect input directory. Please specify --input")
                logger.error("Looking for: organized_matrices/ or directories with atlas structure")
                sys.exit(1)
            logger.info(f"📁 Auto-detected input: {input_path}")
    
    # Define pipeline steps
    steps = {
        '01': {
            'script': 'extract_connectivity_matrices.py',
            'name': 'Connectivity Extraction',
            'args': []  # Will be populated based on args
        },
        '02': {
            'script': 'metric_optimizer.py',
            'name': 'Network Quality Optimization',
            'args': [str(input_path), '--output', str(output_path / 'optimization_results')]
        },
        '03': {
            'script': 'optimal_selection.py', 
            'name': 'Quality-Based Selection',
            'args': [str(output_path / 'optimization_results' / 'optimized_metrics.csv'), 
                    '--output', str(output_path / 'selected_combinations')]
        },
        '04': {
            'script': 'statistical_analysis.py',
            'name': 'Statistical Analysis', 
            'args': [str(output_path / 'selected_combinations' / 'selected_combinations.csv'),
                    '--output', str(output_path / 'statistical_results')]
        }
    }
    
    # Configure step 01 arguments if needed
    if args.step == '01' or args.step == 'all':
        step01_args = [str(data_path)]
        
        # Use JSON config approach if provided
        if args.extraction_config:
            step01_args.extend(['--config', args.extraction_config])
        elif args.config:
            # Fallback to main config if no extraction-specific config
            step01_args.extend(['--config', args.config])
        
        # Add pilot mode if requested
        if args.pilot:
            step01_args.append('--pilot')
            if args.pilot_count:
                step01_args.extend(['--pilot-count', str(args.pilot_count)])
        
        # Set output directory for step 01
        step01_args.extend(['--output', str(output_path / 'organized_matrices')])
        
        steps['01']['args'] = step01_args
    
    # Update step 02 input path if running step 01 first
    if args.step == 'all' and '01' in steps:
        steps['02']['args'][0] = str(output_path / 'organized_matrices')

    # Add config to all steps if provided
    if args.config:
        for step_info in steps.values():
            step_info['args'].extend(['--config', args.config])
    
    # Determine which steps to run
    if args.step == 'all':
        steps_to_run = ['01', '02', '03', '04']
    elif args.step == 'analysis':
        steps_to_run = ['02', '03', '04']
    else:
        steps_to_run = [args.step]
    
    # Run the pipeline steps
    success_count = 0
    total_steps = len(steps_to_run)
    
    for step_num in steps_to_run:
        step_info = steps[step_num]
        
        logger.info(f"\n📊 Step {step_num}: {step_info['name']}")
        logger.info("-" * 50)
        
        success = run_step(
            step_info['script'], 
            step_info['args'], 
            logger, 
            f"Step {step_num}"
        )
        
        if success:
            success_count += 1
        else:
            logger.error(f"Pipeline failed at step {step_num}")
            if args.step == 'all':
                logger.error("Stopping pipeline due to failure")
                break
            else:
                sys.exit(1)
    
    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 Pipeline Summary")
    logger.info("=" * 50)
    logger.info(f"Steps completed: {success_count}/{total_steps}")
    logger.info(f"Output directory: {output_path.absolute()}")
    
    if success_count == total_steps:
        logger.info("✅ Pipeline completed successfully!")
        
        # Show next steps
        logger.info("\n📋 Results Available:")
        if '02' in steps_to_run:
            logger.info(f"  📊 Optimization results: {output_path}/optimization_results/")
        if '03' in steps_to_run:
            logger.info(f"  🎯 Selected combinations: {output_path}/selected_combinations/")
        if '04' in steps_to_run:
            logger.info(f"  📈 Statistical analysis: {output_path}/statistical_results/")
            
    else:
        logger.error("❌ Pipeline completed with errors")
        sys.exit(1)

if __name__ == '__main__':
    main()
