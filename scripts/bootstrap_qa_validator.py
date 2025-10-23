#!/usr/bin/env python3
# Supports: --dry-run (prints intended actions without running)
# When run without arguments the script prints help: parser.print_help()
"""
Bootstrap QA Validator
======================

Optional utility (Extras): Not required for the core OptiConn pipeline
(Steps 01–03). Provides bootstrap validation for QA assessment using
established cross-validation methods. Automatically splits data into
multiple waves for robust QA validation.

Implements         # Create wave configuration
        wave_config = {
            "test_config": {
                "name": f"bootstrap_qa_wave_{wave}",
                "description": f"Bootstrap QA validation - Wave {wave} ({len(subjects)} subjects)"
            },
            "data_selection": {
                "source_dir": data_dir,
                "selection_method": "specific",
                "specific_subjects": subjects,
                "n_subjects": len(subjects),
                "file_pattern": "*.fz"  # Support for DSI Studio .fz files
            }, bootstrap validation for QA assessment using
established cross-validation methods. Automatically splits data into
multiple waves for robust QA validation.

Author: Braingraph Pipeline Team
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import ShuffleSplit

from scripts.utils.runtime import configure_stdio

warnings.filterwarnings("ignore")


def setup_logging():
    """Set up logging configuration."""
    configure_stdio()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_bootstrap_configs(
    data_dir, qa_percentage=0.2, n_waves=2, output_dir="bootstrap_configs"
):
    """
    Create bootstrap validation configurations using established CV methods.

    Args:
        data_dir: Directory containing .fz files
        qa_percentage: Percentage of data to use for each QA wave (default 20%)
        n_waves: Number of bootstrap waves (default 2)
        output_dir: Directory to save configurations
    """

    logging.info(" Creating bootstrap QA validation configs...")
    logging.info(f"   Data directory: {data_dir}")
    logging.info(f"   QA percentage: {qa_percentage*100:.0f}% per wave")
    logging.info(f"   Number of waves: {n_waves}")

    # Find all .fz files
    data_path = Path(data_dir)
    if not data_path.exists():
        logging.error(f" Data directory not found: {data_dir}")
        return False

    fz_files = list(data_path.glob("*.fz"))
    if len(fz_files) == 0:
        logging.error(f" No .fz files found in {data_dir}")
        return False

    logging.info(f" Found {len(fz_files)} .fz files")

    # Calculate sample size per wave
    n_total = len(fz_files)
    n_per_wave = max(3, int(n_total * qa_percentage))  # Minimum 3 subjects per wave

    logging.info(
        f" Sample size per wave: {n_per_wave} subjects ({n_per_wave/n_total*100:.1f}%)"
    )

    # Use ShuffleSplit for robust bootstrap sampling
    cv_splitter = ShuffleSplit(n_splits=n_waves, test_size=n_per_wave, random_state=42)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate file indices
    file_indices = np.arange(len(fz_files))

    bootstrap_configs = []

    for wave_idx, (_, test_indices) in enumerate(cv_splitter.split(file_indices)):
        wave_name = f"bootstrap_qa_wave_{wave_idx + 1}"
        selected_files = [fz_files[i].name for i in test_indices]

        config = {
            "test_config": {
                "name": wave_name,
                "description": f"Bootstrap QA validation - Wave {wave_idx + 1} ({len(selected_files)} subjects)",
            },
            "data_selection": {
                "source_dir": str(data_path),
                "selection_method": "specific",
                "specific_subjects": selected_files,
                "n_subjects": len(selected_files),
            },
            "pipeline_config": {
                "extraction_config": "optimal_config.json",
                "steps_to_run": ["01", "02"],
                "output_base_dir": f"bootstrap_results_{wave_name}",
            },
            "quality_checks": {
                "run_uniqueness_check": True,
                "run_outlier_analysis": True,
                "quality_thresholds": {
                    "min_diversity_score": 0.05,
                    "max_outlier_rate": 0.20,
                },
                "save_detailed_metrics": True,
            },
            "bootstrap_validation": {
                "wave_number": wave_idx + 1,
                "total_waves": n_waves,
                "sampling_method": "ShuffleSplit",
                "qa_percentage": qa_percentage,
                "random_state": 42,
            },
        }

        # Save configuration
        config_file = output_path / f"{wave_name}.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        bootstrap_configs.append(
            {
                "config_file": str(config_file),
                "wave_name": wave_name,
                "subjects": selected_files,
            }
        )

        logging.info(f" Created {wave_name}: {len(selected_files)} subjects")
        logging.info(f"   Config saved: {config_file}")

    # Create master bootstrap configuration
    result_dirs = [
        f"bootstrap_results_{config['wave_name']}" for config in bootstrap_configs
    ]
    comparison_cmd = (
        f"python bootstrap_qa_validator.py validate {' '.join(result_dirs)}"
    )

    master_config = {
        "bootstrap_validation": {
            "method": "ShuffleSplit",
            "qa_percentage": qa_percentage,
            "n_waves": n_waves,
            "total_subjects": n_total,
            "subjects_per_wave": n_per_wave,
            "random_state": 42,
        },
        "waves": bootstrap_configs,
        "execution_commands": [
            f"python run_pipeline.py --test-config {config['config_file']}"
            for config in bootstrap_configs
        ],
        "comparison_command": comparison_cmd,
    }

    master_file = output_path / "bootstrap_master_config.json"
    with open(master_file, "w") as f:
        json.dump(master_config, f, indent=2)

    logging.info("\n Bootstrap configuration summary:")
    logging.info(f"   Total waves: {n_waves}")
    logging.info(f"   Subjects per wave: {n_per_wave}")
    logging.info(f"   Coverage: {n_per_wave * n_waves / n_total * 100:.1f}% of dataset")
    logging.info(f"   Master config: {master_file}")

    return True


def load_bootstrap_results(result_dirs):
    """Load results from multiple bootstrap waves."""

    bootstrap_data = []

    for result_dir in result_dirs:
        agg_file = Path(result_dir) / "aggregated_network_measures.csv"
        if not agg_file.exists():
            logging.warning(f"  Results not found: {agg_file}")
            continue

        df = pd.read_csv(agg_file)
        df["bootstrap_wave"] = Path(result_dir).name
        bootstrap_data.append(df)

        logging.info(f" Loaded wave {Path(result_dir).name}: {len(df)} records")

    if not bootstrap_data:
        logging.error(" No bootstrap results loaded")
        return None

    combined_df = pd.concat(bootstrap_data, ignore_index=True)
    logging.info(f" Combined bootstrap data: {len(combined_df)} total records")

    return combined_df


def compute_bootstrap_qa_metrics(df):
    """Compute QA metrics for each bootstrap wave and overall statistics."""

    if df is None:
        return None

    # Key quality measures
    quality_measures = [
        "density",
        "global_efficiency(binary)",
        "clustering_coeff_average(binary)",
    ]

    bootstrap_results = {
        "wave_metrics": {},
        "stability_analysis": {},
        "bootstrap_confidence": {},
    }

    # Compute metrics for each wave
    for wave in df["bootstrap_wave"].unique():
        wave_data = df[df["bootstrap_wave"] == wave]
        wave_metrics = {}

        for measure in quality_measures:
            if measure not in df.columns:
                continue

            values = wave_data[measure].dropna()
            if len(values) == 0:
                continue

            wave_metrics[f"{measure}_mean"] = np.mean(values)
            wave_metrics[f"{measure}_std"] = np.std(values)
            wave_metrics[f"{measure}_median"] = np.median(values)
            wave_metrics[f"{measure}_iqr"] = np.percentile(values, 75) - np.percentile(
                values, 25
            )

        bootstrap_results["wave_metrics"][wave] = wave_metrics

    # Statistical stability analysis using bootstrap theory
    for measure in quality_measures:
        if measure not in df.columns:
            continue

        # Collect means from each wave
        wave_means = []
        for wave_metrics in bootstrap_results["wave_metrics"].values():
            if f"{measure}_mean" in wave_metrics:
                wave_means.append(wave_metrics[f"{measure}_mean"])

        if len(wave_means) >= 2:
            # Bootstrap statistics
            bootstrap_mean = np.mean(wave_means)
            bootstrap_std = np.std(wave_means, ddof=1)  # Sample standard deviation
            bootstrap_sem = bootstrap_std / np.sqrt(
                len(wave_means)
            )  # Standard error of mean

            # Confidence interval (assuming normal distribution)
            confidence_level = 0.95
            t_value = stats.t.ppf((1 + confidence_level) / 2, len(wave_means) - 1)
            ci_lower = bootstrap_mean - t_value * bootstrap_sem
            ci_upper = bootstrap_mean + t_value * bootstrap_sem

            # Coefficient of variation between waves
            cv_between_waves = (
                bootstrap_std / bootstrap_mean if bootstrap_mean != 0 else float("in")
            )

            bootstrap_results["stability_analysis"][measure] = {
                "bootstrap_mean": bootstrap_mean,
                "bootstrap_std": bootstrap_std,
                "bootstrap_sem": bootstrap_sem,
                "cv_between_waves": cv_between_waves,
                "n_waves": len(wave_means),
                "wave_means": wave_means,
            }

            bootstrap_results["bootstrap_confidence"][measure] = {
                "confidence_level": confidence_level,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_width": ci_upper - ci_lower,
                "relative_ci_width": (
                    (ci_upper - ci_lower) / bootstrap_mean
                    if bootstrap_mean != 0
                    else float("in")
                ),
            }

    return bootstrap_results


def assess_bootstrap_stability(bootstrap_results):
    """Assess QA stability using bootstrap validation criteria."""

    if not bootstrap_results or "stability_analysis" not in bootstrap_results:
        return None

    logging.info("\n BOOTSTRAP QA VALIDATION ASSESSMENT")
    logging.info("=" * 60)

    stability_scores = []

    for measure, stats_data in bootstrap_results["stability_analysis"].items():
        cv = stats_data["cv_between_waves"]

        # Confidence interval analysis
        if measure in bootstrap_results["bootstrap_confidence"]:
            ci_data = bootstrap_results["bootstrap_confidence"][measure]
            rel_ci_width = ci_data["relative_ci_width"]

            logging.info(f"\n {measure}:")
            logging.info(f"   Bootstrap mean: {stats_data['bootstrap_mean']:.4f}")
            logging.info(f"   Between-wave CV: {cv:.4f}")
            logging.info(
                f"   95% CI: [{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}]"
            )
            logging.info(f"   Relative CI width: {rel_ci_width:.4f}")

            # Stability assessment based on established criteria
            if cv < 0.10 and rel_ci_width < 0.20:
                stability = "EXCELLENT"
                score = 4
            elif cv < 0.20 and rel_ci_width < 0.40:
                stability = "GOOD"
                score = 3
            elif cv < 0.35 and rel_ci_width < 0.60:
                stability = "MODERATE"
                score = 2
            else:
                stability = "POOR"
                score = 1

            logging.info(f"   Stability: {stability}")
            stability_scores.append(score)

    # Overall assessment
    if stability_scores:
        avg_score = np.mean(stability_scores)

        if avg_score >= 3.5:
            overall_stability = "EXCELLENT"
            recommendation = (
                " QA metrics are highly stable - proceed with full dataset analysis"
            )
        elif avg_score >= 2.5:
            overall_stability = "GOOD"
            recommendation = " QA metrics are stable - suitable for analysis"
        elif avg_score >= 1.5:
            overall_stability = "MODERATE"
            recommendation = (
                "  Consider increasing QA sample size or additional validation"
            )
        else:
            overall_stability = "POOR"
            recommendation = (
                " QA metrics are unstable - review parameters and increase sample size"
            )

        logging.info("\n OVERALL BOOTSTRAP ASSESSMENT:")
        logging.info(f"   Stability rating: {overall_stability}")
        logging.info(f"   Average stability score: {avg_score:.2f}/4.0")
        logging.info(f"   Number of measures assessed: {len(stability_scores)}")
        logging.info(f"   Recommendation: {recommendation}")

        return {
            "overall_stability": overall_stability,
            "average_score": avg_score,
            "individual_scores": stability_scores,
            "recommendation": recommendation,
            "n_measures": len(stability_scores),
        }

    return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Bootstrap QA Validator with Cross-Validation"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create configs command
    create_parser = subparsers.add_parser(
        "create", help="Create bootstrap validation configurations"
    )
    create_parser.add_argument("data_dir", help="Directory containing .fz files")
    create_parser.add_argument(
        "--qa-percentage",
        type=float,
        default=0.2,
        help="Percentage of data per wave (default: 0.2)",
    )
    create_parser.add_argument(
        "--n-waves", type=int, default=2, help="Number of bootstrap waves (default: 2)"
    )
    create_parser.add_argument(
        "--output-dir", default="bootstrap_configs", help="Output directory for configs"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate bootstrap QA results"
    )
    validate_parser.add_argument(
        "result_dirs", nargs="+", help="Bootstrap result directories"
    )
    validate_parser.add_argument(
        "--output", default="bootstrap_validation_report", help="Output directory"
    )

    # Generate command (for parameter optimization workflow)
    generate_parser = subparsers.add_parser(
        "generate", help="Generate bootstrap data from wave configuration"
    )
    generate_parser.add_argument(
        "--wave-config", required=True, help="Wave configuration JSON file"
    )

    args = parser.parse_args()
    setup_logging()

    if args.command == "create":
        logging.info("  Creating bootstrap QA validation configurations...")
        success = create_bootstrap_configs(
            args.data_dir,
            qa_percentage=args.qa_percentage,
            n_waves=args.n_waves,
            output_dir=args.output_dir,
        )
        return 0 if success else 1

    elif args.command == "validate":
        logging.info(" Running bootstrap QA validation analysis...")

        # Load bootstrap results
        bootstrap_df = load_bootstrap_results(args.result_dirs)
        if bootstrap_df is None:
            return 1

        # Compute bootstrap metrics
        bootstrap_results = compute_bootstrap_qa_metrics(bootstrap_df)
        if bootstrap_results is None:
            return 1

        # Assess stability
        stability_assessment = assess_bootstrap_stability(bootstrap_results)

        # Save detailed report
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        report = {
            "bootstrap_results": bootstrap_results,
            "stability_assessment": stability_assessment,
            "analysis_metadata": {
                "n_waves": len(args.result_dirs),
                "result_directories": args.result_dirs,
                "total_records": len(bootstrap_df),
            },
        }

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        report_converted = convert_numpy(report)

        report_file = output_dir / "bootstrap_qa_validation_report.json"
        with open(report_file, "w") as f:
            json.dump(report_converted, f, indent=2)

        # Also output the assessment to stdout for pipeline parsing
        print(
            json.dumps(
                {"overall_assessment": stability_assessment, "status": "completed"}
            )
        )

        logging.info(f"\n Detailed bootstrap report saved: {report_file}")

        return 0

    elif args.command == "generate":
        logging.info("  Generating bootstrap data from wave configuration...")

        # Load wave configuration
        try:
            with open(args.wave_config, "r") as f:
                wave_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f" Failed to load wave configuration: {e}")
            return 1

        # Extract configuration parameters
        wave_id = wave_config.get("wave_id", "unknown_wave")
        wave_name = wave_config.get("wave_name", "Unknown")
        data_dir = wave_config.get("data_directory")
        parameters = wave_config.get("parameters", {})
        sample_percentage = wave_config.get("sample_percentage", 20)
        output_directory = wave_config.get(
            "output_directory", f"bootstrap_results_{wave_id}"
        )
        output_suffix = wave_config.get("output_suffix", wave_id)

        if not data_dir or not Path(data_dir).exists():
            logging.error(f" Data directory not found: {data_dir}")
            return 1

        logging.info(
            f" Wave: {wave_name} ({wave_config.get('description', 'No description')})"
        )
        logging.info(f" Data directory: {data_dir}")
        logging.info(f" Sample percentage: {sample_percentage}%")
        logging.info(f"  Parameters: {parameters}")

        # Calculate actual number of subjects from percentage
        data_path = Path(data_dir)
        if not data_path.exists():
            logging.error(f" Data directory not found: {data_dir}")
            return 1

        all_files = list(data_path.glob("*.fz"))
        if not all_files:
            logging.error(f" No .fz files found in: {data_dir}")
            return 1

        n_subjects = max(1, int(len(all_files) * sample_percentage / 100))
        logging.info(
            f" Selected {n_subjects} subjects from {len(all_files)} available files"
        )

        # Create temporary test configuration for this wave
        temp_test_config = {
            "test_config": {
                "name": wave_id,
                "description": wave_config.get(
                    "description", f"Bootstrap wave {wave_name}"
                ),
                "enabled": True,
            },
            "data_selection": {
                "source_dir": data_dir,
                "selection_method": "random",
                "n_subjects": n_subjects,  # Use calculated number, not percentage
                "random_seed": hash(wave_id)
                % 1000,  # Deterministic but different per wave
                "file_pattern": "*.fz",
            },
            "pipeline_config": {
                "steps_to_run": ["01", "02", "03", "04"],
                "extraction_config": "sweep_config.json",  # Will be updated with wave parameters
                "output_base_dir": output_directory,
            },
        }

        # Save temporary test configuration
        temp_config_file = f"temp_{wave_id}_config.json"
        with open(temp_config_file, "w") as f:
            json.dump(temp_test_config, f, indent=2)

        logging.info(f" Created temporary config: {temp_config_file}")

        # Create extraction config with wave parameters
        if parameters:
            # Load base extraction config
            base_extraction_config = "sweep_config.json"
            try:
                with open(base_extraction_config, "r") as f:
                    extraction_config = json.load(f)
            except FileNotFoundError:
                # Create minimal extraction config
                extraction_config = {
                    "dsi_studio_cmd": "/usr/local/bin/dsi_studio",
                    "atlases": ["FreeSurferDKT_Cortical"],
                    "connectivity_values": ["count", "fa"],
                }

            # Update with wave parameters - map to DSI Studio parameters
            parameter_mapping = {
                "track_count": "tract_count",
                "step_size": "step_size",
                "angular_threshold": "turning_angle",
                "fa_threshold": "fa_threshold",
            }

            for param_name, param_value in parameters.items():
                dsi_param = parameter_mapping.get(param_name, param_name)
                extraction_config[dsi_param] = param_value

            # Save wave-specific extraction config
            wave_extraction_config = f"extraction_config_{wave_id}.json"
            with open(wave_extraction_config, "w") as f:
                json.dump(extraction_config, f, indent=2)

            # Update temp config to use wave-specific extraction config
            temp_test_config["pipeline_config"][
                "extraction_config"
            ] = wave_extraction_config

            # Re-save temp config with updated extraction config reference
            with open(temp_config_file, "w") as f:
                json.dump(temp_test_config, f, indent=2)

            logging.info(f" Created wave extraction config: {wave_extraction_config}")
            logging.info(
                f"    DSI Studio parameters: {[(k,v) for k,v in extraction_config.items() if k in parameter_mapping.values()]}"
            )

        # Run pipeline with the temporary configuration
        pipeline_cmd = [
            "python",
            "run_pipeline.py",
            "--test-config",
            temp_config_file,
            "--verbose",
        ]

        logging.debug(f" Running pipeline command: {' '.join(pipeline_cmd)}")

        result = subprocess.run(pipeline_cmd)

        if result.returncode == 0:
            logging.info(f" Bootstrap wave {wave_name} completed successfully")
            # Cleanup temporary files on success
            try:
                os.remove(temp_config_file)
                if parameters and f"extraction_config_{wave_id}.json" in locals():
                    os.remove(f"extraction_config_{wave_id}.json")
            except FileNotFoundError:
                pass
        else:
            logging.error(f" Bootstrap wave {wave_name} failed")
            logging.info(
                f" Temporary files preserved for debugging: {temp_config_file}"
            )
            if parameters:
                logging.info(f"    Extraction config: extraction_config_{wave_id}.json")

        return result.returncode

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
