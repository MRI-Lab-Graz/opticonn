#!/usr/bin/env python3
"""
Cross-Validation Bootstrap Optimizer
===================================

Runs parameter optimization across two bootstrap waves to find
optimal DSI Studio parameters using cross-validation.

Author: Braingraph Pipeline Team
"""

import json
import os
import sys
import subprocess
import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np
import random

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def generate_wave_configs(data_dir, output_dir):
    """Generate wave configuration files."""
    
    # Create configs subdirectory
    configs_dir = Path(output_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Wave 1 configuration
    wave1_config = {
        "test_config": {
            "name": "bootstrap_qa_wave_1",
            "description": "Bootstrap QA Wave 1 - Quick test validation"
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": 3,
            "random_seed": 42,
            "file_pattern": "*.fz"
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": "configs/braingraph_default_config.json"
        },
        "bootstrap": {
            "n_iterations": 5,
            "sample_ratio": 0.8
        }
    }
    
    # Wave 2 configuration (different random seed)
    wave2_config = {
        "test_config": {
            "name": "bootstrap_qa_wave_2",
            "description": "Bootstrap QA Wave 2 - Quick test validation"
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": 3,
            "random_seed": 84,
            "file_pattern": "*.fz"
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": "configs/braingraph_default_config.json"
        },
        "bootstrap": {
            "n_iterations": 5,
            "sample_ratio": 0.8
        }
    }
    
    # Save configurations
    wave1_path = configs_dir / "bootstrap_qa_wave_1.json"
    wave2_path = configs_dir / "bootstrap_qa_wave_2.json"
    
    with open(wave1_path, 'w') as f:
        json.dump(wave1_config, f, indent=2)
    
    with open(wave2_path, 'w') as f:
        json.dump(wave2_config, f, indent=2)
    
    logging.info(f"📝 Generated wave configurations in {configs_dir}")
    
    return str(wave1_path), str(wave2_path)

def load_wave_config(config_file):
    """Load wave configuration."""
    with open(config_file, 'r') as f:
        return json.load(f)

def run_wave_pipeline(wave_config_file, output_base_dir):
    """Run pipeline for a single wave."""
    logging.info(f"🚀 Running pipeline for {wave_config_file}")
    
    # Load wave configuration
    wave_config = load_wave_config(wave_config_file)
    wave_name = wave_config['test_config']['name']
    
    logging.info(f"📋 Wave configuration loaded:")
    logging.info(f"   • Name: {wave_name}")
    logging.info(f"   • Data source: {wave_config['data_selection']['source_dir']}")
    logging.info(f"   • Subset size (n_subjects): {wave_config['data_selection'].get('n_subjects')}")
    logging.info(f"   • Bootstrap iterations: {wave_config.get('bootstrap', {}).get('n_iterations')}")
    
    # Create output directory for this wave
    wave_output_dir = Path(output_base_dir) / wave_name
    wave_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Created wave output directory: {wave_output_dir}")

    # List all available files in source and save manifest
    try:
        src_dir = Path(wave_config['data_selection']['source_dir'])
        patterns = [wave_config['data_selection'].get('file_pattern', '*.fz')]
        files = []
        for pat in patterns:
            files.extend(sorted([p for p in src_dir.rglob(pat)]))
        # Also include .fib.gz if not already covered
        files.extend(sorted([p for p in src_dir.rglob('*.fib.gz')]))
        # Deduplicate
        seen = set()
        uniq = []
        for p in files:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        available_manifest = wave_output_dir / 'available_files.txt'
        with available_manifest.open('w') as mf:
            for p in uniq:
                mf.write(str(p) + "\n")
        logging.info(f"📄 Available files listed: {available_manifest} ({len(uniq)})")
    except Exception as e:
        logging.warning(f"⚠️  Could not list available files: {e}")
    
    # Determine selection for this wave
    n_subjects = int(wave_config['data_selection'].get('n_subjects') or 3)
    seed = int(wave_config['data_selection'].get('random_seed') or 42)
    random.seed(seed)
    # Prefer .fz, then .fib.gz
    fz_files = [p for p in uniq if str(p).endswith('.fz')]
    fib_files = [p for p in uniq if str(p).endswith('.fib.gz')]
    pool = fz_files + fib_files
    if not pool:
        logging.error("❌ No candidate files found for selection")
        return False
    if n_subjects >= len(pool):
        selected = pool
    else:
        selected = random.sample(pool, n_subjects)
    # Write selected manifest and build staging dir with symlinks
    selected_manifest = wave_output_dir / 'selected_files.txt'
    with selected_manifest.open('w') as sf:
        for p in selected:
            sf.write(str(p) + "\n")
    logging.info(f"📄 Selected files listed: {selected_manifest} ({len(selected)})")
    staging_dir = wave_output_dir / 'selected_data'
    staging_dir.mkdir(exist_ok=True)
    for p in selected:
        dest = staging_dir / p.name
        try:
            if not dest.exists():
                dest.symlink_to(p)
        except OSError:
            # Fallback to copy if symlink not permitted
            import shutil
            shutil.copy2(p, dest)
    logging.info(f"📂 Staging data directory: {staging_dir}")

    # Run the pipeline for this wave
    cmd = [
        sys.executable, 'run_pipeline.py',
        '--data-dir', str(staging_dir),
        '--step', 'all',
        '--output', str(wave_output_dir),
        '--extraction-config', wave_config.get('pipeline_config', {}).get('extraction_config', 'configs/braingraph_default_config.json'),
        '--verbose'
    ]
    
    logging.info(f"🔧 Command to execute: {' '.join(cmd)}")
    logging.info(f"⏳ Starting pipeline execution for {wave_name}...")
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=0,  # Unbuffered
            universal_newlines=True
        )
        
        # Print output in real-time with immediate flush
        for line in iter(process.stdout.readline, ''):
            if line.strip():  # Only print non-empty lines
                print(f"[{wave_name}] {line.rstrip()}", flush=True)
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            logging.info(f"✅ Wave {wave_name} completed successfully")
            return True
        else:
            logging.error(f"❌ Wave {wave_name} failed with return code {return_code}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Wave {wave_name} failed with exception: {e}")
        return False

def main():
    """Main cross-validation optimizer."""
    parser = argparse.ArgumentParser(description='Cross-Validation Bootstrap Optimizer')
    parser.add_argument('-i', '--data-dir', required=True, help='Data directory')
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--wave1-config', help='Wave 1 configuration file')
    parser.add_argument('--wave2-config', help='Wave 2 configuration file')
    
    args = parser.parse_args()
    
    setup_logging()
    logging.info("🎯 CROSS-VALIDATION BOOTSTRAP OPTIMIZER")
    logging.info("=" * 50)
    logging.info(f"📂 Input data directory: {args.data_dir}")
    logging.info(f"📁 Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Created output directory: {output_dir}")
    
    # Determine wave configurations
    if args.wave1_config and args.wave2_config:
        logging.info("📋 Using provided wave configuration files")
        wave1_config = args.wave1_config
        wave2_config = args.wave2_config
    elif args.config:
        logging.info("📋 Loading master configuration file")
        # Load master config and extract wave configs
        with open(args.config, 'r') as f:
            master_config = json.load(f)
        wave1_config = master_config.get('wave1_config')
        wave2_config = master_config.get('wave2_config')
        
        # If not specified in master config, generate them
        if not wave1_config or not wave2_config:
            logging.info("📝 Generating wave configurations from master config")
            wave1_config, wave2_config = generate_wave_configs(args.data_dir, output_dir)
    else:
        logging.info("📝 Auto-generating default wave configurations")
        # Generate default wave configurations
        wave1_config, wave2_config = generate_wave_configs(args.data_dir, output_dir)
    
    logging.info(f"📁 Output directory: {output_dir}")
    logging.info(f"📄 Wave 1 config: {wave1_config}")
    logging.info(f"📄 Wave 2 config: {wave2_config}")
    
    # Record start time
    start_time = time.time()
    
    # Run Wave 1
    logging.info("\\n" + "🌊" * 20)
    logging.info("🌊 RUNNING WAVE 1")
    logging.info("🌊" * 20)
    wave1_start = time.time()
    wave1_success = run_wave_pipeline(wave1_config, output_dir)
    wave1_duration = time.time() - wave1_start
    logging.info(f"⏱️  Wave 1 completed in {wave1_duration:.1f} seconds")
    
    # Run Wave 2
    logging.info("\\n" + "🌊" * 20)
    logging.info("🌊 RUNNING WAVE 2")
    logging.info("🌊" * 20)
    wave2_start = time.time()
    wave2_success = run_wave_pipeline(wave2_config, output_dir)
    wave2_duration = time.time() - wave2_start
    logging.info(f"⏱️  Wave 2 completed in {wave2_duration:.1f} seconds")
    
    # Final summary
    total_duration = time.time() - start_time
    logging.info("\\n" + "=" * 60)
    
    if wave1_success and wave2_success:
        logging.info("✅ CROSS-VALIDATION COMPLETED SUCCESSFULLY")
        logging.info(f"📊 Results saved in: {output_dir}")
        logging.info(f"⏱️  Total runtime: {total_duration:.1f} seconds")
        logging.info(f"   • Wave 1: {wave1_duration:.1f}s")
        logging.info(f"   • Wave 2: {wave2_duration:.1f}s")
        # Auto-aggregate top candidates across waves
        try:
            import subprocess
            optimization_results_dir = Path(output_dir) / 'optimization_results'
            optimization_results_dir.mkdir(parents=True, exist_ok=True)
            wave1_dir = Path(output_dir) / 'bootstrap_qa_wave_1'
            wave2_dir = Path(output_dir) / 'bootstrap_qa_wave_2'
            cmd = [
                sys.executable, 'scripts/aggregate_wave_candidates.py',
                str(optimization_results_dir), str(wave1_dir), str(wave2_dir)
            ]
            logging.info(f"🧮 Aggregating top candidates across waves → {optimization_results_dir}")
            subprocess.run(cmd, check=True)
            logging.info("📄 top3_candidates.json and all_candidates_ranked.json generated")
        except Exception as e:
            logging.warning(f"⚠️  Failed to aggregate top candidates automatically: {e}")
    else:
        logging.error("❌ CROSS-VALIDATION FAILED")
        logging.error(f"⏱️  Runtime before failure: {total_duration:.1f} seconds")
        if not wave1_success:
            logging.error("   • Wave 1 failed")
        if not wave2_success:
            logging.error("   • Wave 2 failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
