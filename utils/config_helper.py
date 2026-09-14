#!/usr/bin/env python3
"""
OptiConn Configuration Helper
============================

Helper script to create and validate OptiConn configuration files.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def create_basic_config(output_path: Path, quick: bool = False) -> None:
    """Create a basic sweep configuration."""

    if quick:
        config = {
            "description": "Quick test configuration - minimal parameters",
            "dsi_studio_cmd": "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
            "atlases": ["FreeSurferDKT_Cortical"],
            "connectivity_values": ["count", "fa"],
            "tract_count": 100000,
            "thread_count": 4,
            "tracking_parameters": {
                "method": 0,
                "otsu_threshold": 0.5,
                "fa_threshold": 0.1,
                "turning_angle": 0.0,
                "step_size": 0.0,
                "smoothing": 0.0,
                "min_length": 10,
                "max_length": 200,
                "track_voxel_ratio": 3.0,
                "check_ending": 0,
                "random_seed": 0,
                "dt_threshold": 0.1,
            },
            "connectivity_options": {
                "connectivity_type": "pass",
                "connectivity_threshold": 0.001,
                "connectivity_output": "matrix,measure",
            },
            "sweep_parameters": {
                "fa_threshold_range": [0.1, 0.15],
                "min_length_range": [10, 20],
                "track_voxel_ratio_range": [2.0, 3.0],
                "connectivity_threshold_range": [0.001, 0.002],
                "tract_count_range": [100000, 250000],
                "sampling": {"method": "grid", "n_samples": 16, "random_seed": 42},
            },
        }
    else:
        config = {
            "description": "Standard sweep configuration for research",
            "dsi_studio_cmd": "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
            "atlases": ["FreeSurferDKT_Cortical", "FreeSurferSeg"],
            "connectivity_values": ["count", "fa", "qa"],
            "tract_count": 500000,
            "thread_count": 6,
            "tracking_parameters": {
                "method": 0,
                "otsu_threshold": 0.5,
                "fa_threshold": 0.1,
                "turning_angle": 0.0,
                "step_size": 0.0,
                "smoothing": 0.0,
                "min_length": 10,
                "max_length": 200,
                "track_voxel_ratio": 3.0,
                "check_ending": 0,
                "random_seed": 0,
                "dt_threshold": 0.1,
            },
            "connectivity_options": {
                "connectivity_type": "pass",
                "connectivity_threshold": 0.001,
                "connectivity_output": "matrix,measure",
            },
            "sweep_parameters": {
                "fa_threshold_range": [0.05, 0.10, 0.15, 0.20],
                "min_length_range": [10, 15, 20, 30],
                "track_voxel_ratio_range": [2.0, 3.0, 4.0],
                "connectivity_threshold_range": [0.0005, 0.001, 0.002, 0.005],
                "tract_count_range": [250000, 500000, 1000000],
                "sampling": {"method": "grid", "n_samples": 48, "random_seed": 42},
            },
        }

    # Write configuration
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Created {'quick' if quick else 'standard'} configuration: {output_path}")


def validate_config(config_path: Path) -> bool:
    """Validate a configuration file."""

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ File not found: {config_path}")
        return False

    # Check required fields
    required_fields = [
        "atlases",
        "connectivity_values",
        "tracking_parameters",
        "connectivity_options",
        "sweep_parameters",
    ]

    missing = []
    for field in required_fields:
        if field not in config:
            missing.append(field)

    if missing:
        print(f"❌ Missing required fields: {missing}")
        return False

    # Check sweep parameters
    sweep = config["sweep_parameters"]
    param_fields = [
        "fa_threshold_range",
        "min_length_range",
        "track_voxel_ratio_range",
        "connectivity_threshold_range",
        "tract_count_range",
    ]

    found_params = 0
    for field in param_fields:
        if field in sweep:
            found_params += 1

    if found_params == 0:
        print("❌ No sweep parameter ranges found")
        return False

    # Estimate number of combinations
    total_combos = 1
    for field in param_fields:
        if field in sweep and isinstance(sweep[field], list):
            total_combos *= len(sweep[field])

    print(f"✅ Configuration is valid")
    print(f"📊 Estimated parameter combinations: {total_combos}")
    print(f"🎯 Found {found_params} parameter ranges")

    # Check for common issues
    warnings = []

    if total_combos > 100:
        warnings.append(
            f"Large parameter grid ({total_combos} combinations) - consider using random sampling"
        )

    if "dsi_studio_cmd" not in config:
        warnings.append("No dsi_studio_cmd specified - will use environment variable")

    tract_counts = sweep.get("tract_count_range", [])
    if isinstance(tract_counts, list) and any(tc > 5000000 for tc in tract_counts):
        warnings.append(
            "Very high tract counts detected - may require significant memory"
        )

    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  • {warning}")

    return True


def show_config_summary(config_path: Path) -> None:
    """Show a summary of the configuration."""

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        return

    print(f"📋 Configuration Summary: {config_path.name}")
    print("=" * 50)
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Atlases: {', '.join(config.get('atlases', []))}")
    print(f"Connectivity values: {', '.join(config.get('connectivity_values', []))}")
    print(f"Base tract count: {config.get('tract_count', 'N/A'):,}")

    sweep = config.get("sweep_parameters", {})
    print("\nSweep Parameters:")
    for param, values in sweep.items():
        if param.endswith("_range") and isinstance(values, list):
            print(f"  {param}: {len(values)} values ({min(values)} to {max(values)})")

    sampling = sweep.get("sampling", {})
    method = sampling.get("method", "grid")
    print(f"\nSampling: {method}")
    if method != "grid":
        print(f"Samples: {sampling.get('n_samples', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="OptiConn Configuration Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a quick test configuration
  python config_helper.py create --output my_config.json --quick
  
  # Create a standard configuration
  python config_helper.py create --output my_config.json
  
  # Validate an existing configuration  
  python config_helper.py validate --config my_config.json
  
  # Show configuration summary
  python config_helper.py summary --config my_config.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new configuration")
    create_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output configuration file"
    )
    create_parser.add_argument(
        "--quick", action="store_true", help="Create quick test configuration"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Configuration file to validate",
    )

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show configuration summary")
    summary_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Configuration file to summarize",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if args.command == "create":
        create_basic_config(args.output, args.quick)

    elif args.command == "validate":
        is_valid = validate_config(args.config)
        return 0 if is_valid else 1

    elif args.command == "summary":
        show_config_summary(args.config)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
