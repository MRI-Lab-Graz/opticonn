#!/usr/bin/env python3
"""
Review Optimization Results
===========================

Provides a non-interactive summary of the results from an optimization run.

Author: Braingraph Pipeline Team
"""

import json
import logging
from pathlib import Path
import argparse
import sys
import pandas as pd

from scripts.utils.runtime import configure_stdio

logger = logging.getLogger(__name__)


def find_results_file(output_dir: Path) -> Path | None:
    """Find the main results JSON file in an output directory."""
    bayesian_results = output_dir / "bayesian_optimization_results.json"
    if bayesian_results.exists():
        return bayesian_results

    sweep_results = output_dir / "sweep_results.json"
    if sweep_results.exists():
        return sweep_results

    return None


def display_summary(results_path: Path):
    """
    Loads results and displays a formatted summary of the best parameters.
    """
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f" Could not read or parse results file: {results_path}")
        logger.error(f"   Error: {e}")
        return

    method = results.get("optimization_method", "unknown")
    best_score = results.get("best_qa_score")

    if method == "bayesian":
        best_params = results.get("best_parameters", {})
        # For bayesian, we need to find the parameter space definition
        # It's not explicitly stored in a user-friendly way, so we'll just show the values.
    elif method == "sweep":
        best_combo = results.get("best_combination", {})
        best_params = best_combo.get("combination", {})
    else:
        logger.error(f" Unknown optimization method '{method}' in results file.")
        return

    if not best_params:
        logger.error(" No best parameters found in the results file.")
        return

    print("\n" + f" Optimization Review: {results_path.parent.name}")
    print("=" * 60)
    print(f"Method: {method.capitalize()}")
    if best_score is not None:
        print(f"Best QA Score: {best_score:.4f}")
    print("\n" + "Chosen Optimal Parameters:")
    print("-" * 60)

    # Create a DataFrame for nice formatting
    summary_data = []
    for param, value in best_params.items():
        summary_data.append({"Parameter": param, "Chosen Value": value})

    df = pd.DataFrame(summary_data)
    df_string = df.to_string(index=False, justify="left")

    print(df_string)
    print("-" * 60)

    # Provide guidance on where to find the full config
    if method == "bayesian":
        print(
            "\n The full configuration can be reconstructed from the parameters above."
        )
        print("   The best iteration's output is in the 'iterations' subdirectory.")
    elif method == "sweep":
        best_config_path = best_combo.get("config_path")
        if best_config_path:
            print("\n The full configuration for this run is stored in:")
            print(f"   {best_config_path}")

    print("=" * 60)


def main():
    """Command line interface for reviewing optimization results."""
    parser = argparse.ArgumentParser(
        description="Review optimization results from a previous run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="The output directory of the 'find-optimal-parameters' run to review.",
    )
    parser.add_argument(
        "--no-emoji", action="store_true", help="Disable emoji in console output"
    )

    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        logger.error(f" Output directory not found: {output_dir}")
        sys.exit(1)

    results_file = find_results_file(output_dir)
    if not results_file:
        logger.error(f" No optimization results file found in {output_dir}")
        logger.error(
            "   (Looking for 'bayesian_optimization_results.json' or 'sweep_results.json')"
        )
        sys.exit(1)

    display_summary(results_file)
    sys.exit(0)


if __name__ == "__main__":
    main()
