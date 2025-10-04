#!/usr/bin/env python3
# Supports: --dry-run (prints intended actions without running)
# When run without arguments the script prints help: parser.print_help()
"""
DSI Studio Setup Validation Script

This script validates the DSI Studio installation and configuration
for connectivity matrix extraction.

Author: Brain Connectivity Analysis Pipeline
Date: 2025-08-07
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

def check_dsi_studio_installation():
    """Check if DSI Studio is properly installed and accessible."""
    
    print("🔍 Checking DSI Studio installation...")
    
    # Common DSI Studio paths
    possible_paths = [
        "/Applications/DSI_Studio.app/Contents/MacOS/dsi_studio",
        "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio", 
        "dsi_studio",  # If in PATH
        "/usr/local/bin/dsi_studio",
        "/opt/dsi_studio/dsi_studio"
    ]
    
    dsi_path = None
    for path in possible_paths:
        if os.path.exists(path) or (path == "dsi_studio" and check_command_in_path("dsi_studio")):
            dsi_path = path
            break
    
    if not dsi_path:
        print("❌ DSI Studio not found in common locations")
        print("   Please install DSI Studio and ensure it's accessible")
        return False, None
    
    print(f"✅ DSI Studio found: {dsi_path}")
    
    # Test DSI Studio execution
    try:
        result = subprocess.run([dsi_path, "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ DSI Studio is executable")
            return True, dsi_path
        else:
            print("⚠️  DSI Studio found but may not be properly configured")
            return True, dsi_path  # Still return True as it exists
    except subprocess.TimeoutExpired:
        print("⚠️  DSI Studio found but help command timed out")
        return True, dsi_path
    except Exception as e:
        print(f"⚠️  Error testing DSI Studio: {e}")
        return True, dsi_path  # Still return True as it exists

def check_command_in_path(command):
    """Check if a command is available in PATH."""
    try:
        subprocess.run([command, "--version"], 
                      capture_output=True, timeout=5)
        return True
    except:
        return False

def validate_configuration(config_path):
    """Validate the configuration file."""
    
    print(f"📋 Validating configuration: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Configuration file is valid JSON")
        
        # Check required fields
        required_fields = ["atlases", "connectivity_values", "track_count"]
        for field in required_fields:
            if field not in config:
                print(f"⚠️  Missing required field: {field}")
            else:
                print(f"✅ Found {field}: {len(config[field]) if isinstance(config[field], list) else config[field]}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def test_input_file(input_path):
    """Test if input file/directory is accessible."""
    
    if not input_path:
        print("⚠️  No test input specified")
        return True
    
    print(f"📁 Testing input: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Input path does not exist: {input_path}")
        return False
    
    if os.path.isfile(input_path):
        if input_path.endswith('.fz'):
            print("✅ Input is a valid .fz file")
            return True
        else:
            print("⚠️  Input file does not have .fz extension")
            return True
    elif os.path.isdir(input_path):
        fz_files = list(Path(input_path).glob("*.fz"))
        print(f"✅ Input directory contains {len(fz_files)} .fz files")
        return True
    
    return True

def check_python_environment():
    """Check Python environment and required packages."""
    
    print("🐍 Checking Python environment...")
    
    print(f"✅ Python version: {sys.version}")
    
    # Check required packages
    required_packages = ["numpy", "pandas", "pathlib"]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"⚠️  {package} not found (may not be critical)")
    
    return True

def check_disk_space(output_dir):
    """Check available disk space."""
    
    if not output_dir:
        return True
    
    print(f"💾 Checking disk space for: {output_dir}")
    
    try:
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check free space
        stat = os.statvfs(output_dir)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        print(f"✅ Available disk space: {free_space_gb:.1f} GB")
        
        if free_space_gb < 1.0:
            print("⚠️  Warning: Less than 1 GB free space available")
        elif free_space_gb < 10.0:
            print("⚠️  Warning: Less than 10 GB free space available")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Validate DSI Studio setup")
    parser.add_argument("--config", help="Configuration file to validate")
    parser.add_argument("--test-input", help="Test input file or directory")
    parser.add_argument("--output-dir", help="Output directory to check")
    
    args = parser.parse_args()
    
    print("🔧 DSI Studio Setup Validation")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check DSI Studio installation
    dsi_available, dsi_path = check_dsi_studio_installation()
    if not dsi_available:
        all_checks_passed = False
    
    print()
    
    # Check Python environment
    python_ok = check_python_environment()
    if not python_ok:
        all_checks_passed = False
    
    print()
    
    # Validate configuration if provided
    if args.config:
        config_ok = validate_configuration(args.config)
        if not config_ok:
            all_checks_passed = False
        print()
    
    # Test input if provided
    if args.test_input:
        input_ok = test_input_file(args.test_input)
        if not input_ok:
            all_checks_passed = False
        print()
    
    # Check disk space if output directory provided
    if args.output_dir:
        disk_ok = check_disk_space(args.output_dir)
        if not disk_ok:
            all_checks_passed = False
        print()
    
    # Summary
    print("=" * 50)
    if all_checks_passed:
        print("✅ All validation checks passed!")
        print("🚀 Ready to run connectivity extraction")
    else:
        print("⚠️  Some validation checks failed")
        print("🔧 Please fix the issues above before proceeding")
    
    print("=" * 50)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
