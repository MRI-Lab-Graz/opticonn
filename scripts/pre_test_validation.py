#!/usr/bin/env python3
"""
Pre-test validation script to ensure environment is ready.
Run this before any testing to catch common setup issues.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_environment():
    """Check if we're in the correct virtual environment."""
    print("🔍 Checking Python environment...")
    
    # Check if we're in virtual environment
    venv_path = os.environ.get('VIRTUAL_ENV')
    if not venv_path:
        print("❌ Virtual environment not activated!")
        print("💡 Run: source braingraph_pipeline/bin/activate")
        return False
    
    # Check if it's the correct venv
    expected_venv = "braingraph_pipeline"
    if expected_venv not in venv_path:
        print(f"⚠️  Different virtual environment active: {venv_path}")
        print(f"💡 Expected: {expected_venv}")
        return False
    
    print(f"✅ Virtual environment active: {venv_path}")
    return True

def check_working_directory():
    """Check if we're in the correct working directory."""
    print("\n🔍 Checking working directory...")
    
    current_dir = Path.cwd()
    expected_files = ['braingraph.py', 'run_pipeline.py', 'scripts', 'configs']
    
    missing_files = []
    for file in expected_files:
        if not (current_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Not in braingraph-pipeline directory!")
        print(f"💡 Missing: {', '.join(missing_files)}")
        print(f"💡 Current directory: {current_dir}")
        print("💡 Run: cd /Volumes/Work/github/braingraph-pipeline")
        return False
    
    print(f"✅ Working directory correct: {current_dir}")
    return True

def check_dsi_studio():
    """Check if DSI Studio is accessible."""
    print("\n🔍 Checking DSI Studio...")
    
    dsi_path = "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio"
    
    if not Path(dsi_path).exists():
        print("❌ DSI Studio not found!")
        print(f"💡 Expected path: {dsi_path}")
        return False
    
    # Check if file is executable (don't run it as it launches GUI)
    if os.access(dsi_path, os.X_OK):
        print("✅ DSI Studio found and executable")
        print(f"   Path: {dsi_path}")
        return True
    else:
        print("❌ DSI Studio found but not executable")
        return False

def check_script_imports():
    """Check if key scripts can be imported."""
    print("\n🔍 Checking script imports...")
    
    # Add current directory to Python path for imports
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    scripts_to_test = [
        'scripts.cross_validation_bootstrap_optimizer',
        'scripts.validate_setup',
        'scripts.extract_connectivity_matrices'
    ]
    
    all_good = True
    for script in scripts_to_test:
        try:
            __import__(script)
            print(f"✅ {script}")
        except ImportError as e:
            print(f"❌ {script}: {e}")
            all_good = False
        except Exception as e:
            print(f"⚠️  {script}: {e}")
            all_good = False
    
    return all_good

def check_configs():
    """Check if main configuration files exist and are valid JSON."""
    print("\n🔍 Checking configuration files...")
    
    configs_to_check = [
        'configs/01_working_config.json',
        'configs/production_config.json',
        'configs/sweep_config.json'
    ]
    
    all_good = True
    for config_path in configs_to_check:
        if not Path(config_path).exists():
            print(f"❌ Missing: {config_path}")
            all_good = False
            continue
        
        try:
            import json
            with open(config_path, 'r') as f:
                json.load(f)
            print(f"✅ {config_path}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {config_path}: {e}")
            all_good = False
    
    return all_good

def main():
    """Run all pre-test checks."""
    print("🚀 Running Pre-Test Environment Validation\n")
    
    checks = [
        ("Environment", check_environment),
        ("Working Directory", check_working_directory),
        ("DSI Studio", check_dsi_studio),
        ("Script Imports", check_script_imports),
        ("Configuration Files", check_configs)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "="*50)
    print("📋 PRE-TEST VALIDATION SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print("="*50)
    if all_passed:
        print("🎉 All checks passed! Ready for testing.")
        return 0
    else:
        print("🚨 Some checks failed. Fix issues before testing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
