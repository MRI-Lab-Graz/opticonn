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
    
    # Create output directory for this wave
    wave_output_dir = Path(output_base_dir) / wave_name
    wave_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the pipeline for this wave
    cmd = [
        sys.executable, 'run_pipeline.py',
        '--data-dir', wave_config['data_selection']['source_dir'],
        '--step', 'all',
        '--output', str(wave_output_dir),
        '--extraction-config', 'configs/01_working_config.json',
        '--verbose'
    ]
    
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"✅ Wave {wave_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Wave {wave_name} failed: {e}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        return False

def main():
    """Main cross-validation optimizer."""
    parser = argparse.ArgumentParser(description='Cross-Validation Bootstrap Optimizer')
    parser.add_argument('--data-dir', required=True, help='Data directory')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--wave1-config', help='Wave 1 configuration file')
    parser.add_argument('--wave2-config', help='Wave 2 configuration file')
    
    args = parser.parse_args()
    
    setup_logging()
    logging.info("🎯 CROSS-VALIDATION BOOTSTRAP OPTIMIZER")
    logging.info("=" * 50)
    
    # Determine wave configurations
    if args.wave1_config and args.wave2_config:
        wave1_config = args.wave1_config
        wave2_config = args.wave2_config
    elif args.config:
        # Load master config and extract wave configs
        with open(args.config, 'r') as f:
            master_config = json.load(f)
        wave1_config = master_config.get('wave1_config', 'cross_validation_configs/bootstrap_qa_wave_1.json')
        wave2_config = master_config.get('wave2_config', 'cross_validation_configs/bootstrap_qa_wave_2.json')
    else:
        # Default to the bootstrap configs we just created
        wave1_config = 'cross_validation_configs/bootstrap_qa_wave_1.json'
        wave2_config = 'cross_validation_configs/bootstrap_qa_wave_2.json'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"📁 Output directory: {output_dir}")
    logging.info(f"📄 Wave 1 config: {wave1_config}")
    logging.info(f"📄 Wave 2 config: {wave2_config}")
    
    # Run Wave 1
    logging.info("\\n🌊 RUNNING WAVE 1")
    wave1_success = run_wave_pipeline(wave1_config, output_dir)
    
    # Run Wave 2
    logging.info("\\n🌊 RUNNING WAVE 2")
    wave2_success = run_wave_pipeline(wave2_config, output_dir)
    
    if wave1_success and wave2_success:
        logging.info("\\n✅ CROSS-VALIDATION COMPLETED SUCCESSFULLY")
        logging.info("📊 Results saved in: {output_dir}")
    else:
        logging.error("\\n❌ CROSS-VALIDATION FAILED")
        sys.exit(1)

if __name__ == '__main__':
    main()
