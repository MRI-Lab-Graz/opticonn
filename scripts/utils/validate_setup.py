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

    print(" Checking DSI Studio installation...")

    # Common DSI Studio paths
    possible_paths = [
        "/Applications/DSI_Studio.app/Contents/MacOS/dsi_studio",
        "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
        "dsi_studio",  # If in PATH
        "/usr/local/bin/dsi_studio",
        "/opt/dsi_studio/dsi_studio",
    ]

    dsi_path = None
    for path in possible_paths:
        if os.path.exists(path) or (
            path == "dsi_studio" and check_command_in_path("dsi_studio")
        ):
            dsi_path = path
            break

    if not dsi_path:
        print(" DSI Studio not found in common locations")
        print("   Please install DSI Studio and ensure it's accessible")
        return False, None

    print(f" DSI Studio found: {dsi_path}")
    # Only check executable bit, do not run DSI Studio
    if not (os.access(dsi_path, os.X_OK) or dsi_path == "dsi_studio"):
        print(f" DSI Studio path is not executable: {dsi_path}")
        return False, dsi_path
    print(" DSI Studio is marked as executable (not launched)")
    return True, dsi_path


def check_command_in_path(command):
    """Check if a command is available in PATH."""
    try:
        subprocess.run([command, "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def validate_configuration(config_path):
    """Validate the configuration file."""

    print(f" Validating configuration: {config_path}")

    if not os.path.exists(config_path):
        print(f" Configuration file not found: {config_path}")
        return False

    # Use json_validator.py for schema/structure validation
    import subprocess

    validator_script = str(Path(__file__).parent / "json_validator.py")
    result = subprocess.run(
        [sys.executable, validator_script, config_path, "--suggest-fixes", "--dry-run"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(" Schema/structure validation failed.")
        return False

    # Value checks
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(" Configuration file is valid JSON")

        # Check required fields
        required_fields = ["atlases", "connectivity_values", "tract_count"]
        for field in required_fields:
            if field not in config:
                print(f"  Missing required field: {field}")
            else:
                val = config[field]
                if isinstance(val, list) and not val:
                    print(f"  Field '{field}' is an empty list")
                else:
                    print(
                        f" Found {field}: {len(val) if isinstance(val, list) else val}"
                    )

        # Check paths
        for key in ["dsi_studio_cmd", "extraction_config"]:
            if key in config:
                path_val = config[key]
                if not os.path.exists(path_val):
                    print(f"  Path for '{key}' does not exist: {path_val}")
                else:
                    print(f" Path for '{key}' exists: {path_val}")

        # Check numbers
        if "tract_count" in config:
            tc = config["tract_count"]
            if not isinstance(tc, int) or tc < 1000:
                print(f"  tract_count should be integer >= 1000 (got {tc})")
            elif tc > 10000000:
                print("  tract_count over 10 million may cause memory issues")
            else:
                print(f" tract_count is in reasonable range: {tc}")
        if "thread_count" in config:
            th = config["thread_count"]
            if not isinstance(th, int) or th < 1 or th > 32:
                print(f"  thread_count should be integer between 1 and 32 (got {th})")
            else:
                print(f" thread_count is in reasonable range: {th}")

        return True

    except json.JSONDecodeError as e:
        print(f" Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f" Error reading configuration: {e}")
        return False


def test_input_file(input_path):
    """Test if input file/directory is accessible."""

    if not input_path:
        print("  No test input specified")
        return True

    print(f" Testing input: {input_path}")

    if not os.path.exists(input_path):
        print(f" Input path does not exist: {input_path}")
        return False

    def check_fib_header(fib_path):
        try:
            with open(fib_path, "rb") as f:
                header = f.read(32)
                if b"DSI Studio Fib" in header:
                    print(f" .fib file header OK: {fib_path}")
                    return True
                else:
                    print(f"  .fib file header missing or invalid: {fib_path}")
                    return False
        except Exception as e:
            print(f" Error reading .fib file: {fib_path} ({e})")
            return False

    def warn_small_file(file_path, min_size_kb=10):
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb < min_size_kb:
            print(f"  File is suspiciously small ({size_kb:.1f} KB): {file_path}")

    if os.path.isfile(input_path):
        ext = Path(input_path).suffix.lower()
        if ext == ".fz":
            warn_small_file(input_path)
            print(" Input is a valid .fz file")
            return True
        elif ext == ".fib":
            warn_small_file(input_path)
            return check_fib_header(input_path)
        else:
            print(f"  Input file does not have .fz or .fib extension: {input_path}")
            return True
    elif os.path.isdir(input_path):
        fz_files = list(Path(input_path).glob("*.fz"))
        fib_files = list(Path(input_path).glob("*.fib"))
        print(
            f" Input directory contains {len(fz_files)} .fz files and {len(fib_files)} .fib files"
        )
        all_ok = True
        for f in fz_files:
            warn_small_file(f)
        for f in fib_files:
            warn_small_file(f)
            if not check_fib_header(f):
                all_ok = False
        if len(fz_files) + len(fib_files) == 0:
            print(" No .fz or .fib files found in input directory!")
            return False
        return all_ok

    return True


def check_python_environment():
    """Check Python environment and required packages."""

    print(" Checking Python environment...")

    print(f" Python version: {sys.version}")

    # Check required packages
    required_packages = ["numpy", "pandas", "pathlib"]

    for package in required_packages:
        try:
            __import__(package)
            print(f" {package} is available")
        except ImportError:
            print(f"  {package} not found (may not be critical)")

    return True


def check_disk_space(output_dir):
    """Check available disk space."""

    if not output_dir:
        return True

    print(f" Checking disk space for: {output_dir}")

    try:
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Check free space
        stat = os.statvfs(output_dir)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        print(f" Available disk space: {free_space_gb:.1f} GB")

        if free_space_gb < 1.0:
            print("  Warning: Less than 1 GB free space available")
        elif free_space_gb < 10.0:
            print("  Warning: Less than 10 GB free space available")

        return True

    except Exception as e:
        print(f"  Could not check disk space: {e}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate DSI Studio setup")
    parser.add_argument("--config", help="Configuration file to validate")
    parser.add_argument("--test-input", help="Test input file or directory")
    parser.add_argument("--output-dir", help="Output directory to check")
    parser.add_argument(
        "--no-input-test",
        action="store_true",
        help="Skip automatic input file validation",
    )

    args = parser.parse_args()

    print(" DSI Studio Setup Validation")
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

    # Automatically test input unless opted out
    auto_input_path = args.test_input
    if not args.no_input_test:
        # If no --test-input provided, try to infer from config
        if not auto_input_path and args.config:
            try:
                with open(args.config, "r") as f:
                    config = json.load(f)
                # Try common keys for input directory
                for key in ["data_dir", "input_dir", "source_dir"]:
                    if key in config:
                        auto_input_path = config[key]
                        print(
                            f" Auto-detected input path from config: {auto_input_path}"
                        )
                        break
            except Exception:
                pass
        if auto_input_path:
            input_ok = test_input_file(auto_input_path)
            if not input_ok:
                all_checks_passed = False
            print()
        else:
            print(
                "  No input path provided or detected; skipping input file validation."
            )
            print()

    # Check output directory writability and disk space
    if args.output_dir:
        # Check if directory exists and is writable
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            test_file = Path(args.output_dir) / ".write_test.tmp"
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            print(f" Output directory is writable: {args.output_dir}")
        except Exception as e:
            print(f" Output directory is not writable: {args.output_dir} ({e})")
            all_checks_passed = False
        # Warn if directory contains important files
        existing_files = list(Path(args.output_dir).glob("*"))
        if existing_files:
            print(
                f"  Output directory is not empty and contains {len(existing_files)} files. Check for possible overwrites."
            )
        disk_ok = check_disk_space(args.output_dir)
        if not disk_ok:
            all_checks_passed = False
        print()

    # Summary
    print("=" * 50)
    if all_checks_passed:
        print(" All validation checks passed!")
        print(" Ready to run connectivity extraction")
    else:
        print("  Some validation checks failed")
        print(" Please fix the issues above before proceeding")

    print("=" * 50)

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
