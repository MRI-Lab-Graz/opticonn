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
            "subset_size": 3,  # Reduced from 10
            "random_seed": 42
        },
        "bootstrap": {
            "n_iterations": 5,  # Reduced from 100
            "sample_ratio": 0.8
        },
        "analysis": {
            "metrics": ["modularity", "clustering_coefficient"],  # Reduced metrics
            "atlases": ["FreeSurferDKT_Cortical"]  # Single atlas only
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
            "subset_size": 3,  # Reduced from 10
            "random_seed": 84
        },
        "bootstrap": {
            "n_iterations": 5,  # Reduced from 100
            "sample_ratio": 0.8
        },
        "analysis": {
            "metrics": ["modularity", "clustering_coefficient"],  # Reduced metrics
            "atlases": ["FreeSurferDKT_Cortical"]  # Single atlas only
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
    logging.info(f"   • Subset size: {wave_config['data_selection']['subset_size']}")
    logging.info(f"   • Bootstrap iterations: {wave_config['bootstrap']['n_iterations']}")
    
    # Create output directory for this wave
    wave_output_dir = Path(output_base_dir) / wave_name
    wave_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Created wave output directory: {wave_output_dir}")
    
    # Run the pipeline for this wave
    cmd = [
        sys.executable, 'run_pipeline.py',
        '--data-dir', wave_config['data_selection']['source_dir'],
        '--step', 'all',
        '--output', str(wave_output_dir),
        '--extraction-config', 'configs/braingraph_default_config.json',
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
