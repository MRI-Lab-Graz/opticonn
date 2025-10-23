#!/usr/bin/env python3
"""
QA Cross-Validation Comparator
==============================

Optional utility (Extras): Not required for the core OptiConn pipeline
(Steps 01–03). Compares QA results between two random subsets to validate
the reliability and stability of quality assessment metrics.

Author: Braingraph Pipeline Team
"""

import pandas as pd
import numpy as np
import json
import sys
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_qa_results(results_dir):
    """Load QA results from a pipeline run."""

    # Load aggregated network measures
    agg_file = Path(results_dir) / "aggregated_network_measures.csv"
    if not agg_file.exists():
        logging.error(f" Aggregated results not found: {agg_file}")
        return None

    df = pd.read_csv(agg_file)
    logging.info(f" Loaded {len(df)} records from {results_dir}")

    return df


def compute_qa_metrics(df):
    """Compute standardized QA metrics from network measures."""

    if df is None or len(df) == 0:
        return None

    # Key quality measures
    quality_measures = [
        "density",
        "global_efficiency(binary)",
        "clustering_coeff_average(binary)",
    ]

    qa_summary = {}

    for measure in quality_measures:
        if measure not in df.columns:
            continue

        values = df[measure].dropna()
        if len(values) == 0:
            continue

        # Compute QA metrics
        qa_summary[f"{measure}_mean"] = np.mean(values)
        qa_summary[f"{measure}_std"] = np.std(values)
        qa_summary[f"{measure}_cv"] = np.std(values) / (
            np.mean(values) + 1e-10
        )  # Coefficient of variation
        qa_summary[f"{measure}_q25"] = np.percentile(values, 25)
        qa_summary[f"{measure}_q75"] = np.percentile(values, 75)
        qa_summary[f"{measure}_range"] = np.max(values) - np.min(values)

        # Outlier rate
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        outliers = ((values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)).sum()
        qa_summary[f"{measure}_outlier_rate"] = outliers / len(values)

    # Parameter diversity
    n_subjects = df["subject_id"].nunique()
    n_atlases = df["atlas"].nunique()
    n_metrics = df["connectivity_metric"].nunique()

    qa_summary["n_subjects"] = n_subjects
    qa_summary["n_atlases"] = n_atlases
    qa_summary["n_connectivity_metrics"] = n_metrics
    qa_summary["total_combinations"] = len(df)
    qa_summary["combinations_per_subject"] = (
        len(df) / n_subjects if n_subjects > 0 else 0
    )

    return qa_summary


def compare_qa_metrics(qa_a, qa_b, set_a_name="Set A", set_b_name="Set B"):
    """Compare QA metrics between two sets."""

    if qa_a is None or qa_b is None:
        logging.error(" Cannot compare - missing QA data")
        return None

    logging.info(f"\n QA Cross-Validation Comparison: {set_a_name} vs {set_b_name}")
    logging.info("=" * 70)

    # Find common metrics
    common_metrics = set(qa_a.keys()) & set(qa_b.keys())

    comparison_results = {}

    for metric in sorted(common_metrics):
        val_a = qa_a[metric]
        val_b = qa_b[metric]

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            # Calculate relative difference
            if val_a != 0:
                rel_diff = abs(val_a - val_b) / abs(val_a)
            elif val_b != 0:
                rel_diff = abs(val_a - val_b) / abs(val_b)
            else:
                rel_diff = 0

            comparison_results[metric] = {
                "set_a": val_a,
                "set_b": val_b,
                "abs_diff": abs(val_a - val_b),
                "rel_diff": rel_diff,
                "agreement": (
                    "Good"
                    if rel_diff < 0.2
                    else "Moderate" if rel_diff < 0.5 else "Poor"
                ),
            }

            logging.info(
                f" {metric:<35}: {val_a:.4f} vs {val_b:.4f} (rel_diff: {rel_diff:.3f})"
            )

    # Overall assessment
    rel_diffs = [comp["rel_diff"] for comp in comparison_results.values()]
    avg_rel_diff = np.mean(rel_diffs) if rel_diffs else float("inf")

    logging.info(f"\n Overall QA Stability Assessment:")
    logging.info(f"   Average relative difference: {avg_rel_diff:.3f}")

    if avg_rel_diff < 0.15:
        stability = "EXCELLENT"
        recommendation = " QA metrics are highly stable - proceed with confidence"
    elif avg_rel_diff < 0.30:
        stability = "GOOD"
        recommendation = " QA metrics are reasonably stable - suitable for analysis"
    elif avg_rel_diff < 0.50:
        stability = "MODERATE"
        recommendation = "  QA metrics show some variability - consider larger sample"
    else:
        stability = "POOR"
        recommendation = (
            " QA metrics are unstable - increase sample size or check parameters"
        )

    logging.info(f"   Stability rating: {stability}")
    logging.info(f"   Recommendation: {recommendation}")

    return {
        "comparison_results": comparison_results,
        "overall_stability": stability,
        "avg_relative_difference": avg_rel_diff,
        "recommendation": recommendation,
    }


def generate_comparison_report(comparison, output_dir):
    """Generate a detailed comparison report."""

    if comparison is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save detailed results to JSON
    report_file = output_dir / "qa_cross_validation_report.json"
    with open(report_file, "w") as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_compatible = {}
        for key, value in comparison.items():
            if isinstance(value, dict):
                json_compatible[key] = {
                    k: float(v) if isinstance(v, np.floating) else v
                    for k, v in value.items()
                }
            else:
                json_compatible[key] = (
                    float(value) if isinstance(value, np.floating) else value
                )

        json.dump(json_compatible, f, indent=2)

    logging.info(f" Detailed report saved: {report_file}")

    # Create summary visualization if matplotlib available
    try:
        comp_results = comparison["comparison_results"]
        metrics = list(comp_results.keys())
        rel_diffs = [comp_results[m]["rel_diff"] for m in metrics]

        plt.figure(figsize=(12, 8))
        colors = [
            "green" if rd < 0.2 else "orange" if rd < 0.5 else "red" for rd in rel_diffs
        ]

        plt.barh(range(len(metrics)), rel_diffs, color=colors, alpha=0.7)
        plt.yticks(range(len(metrics)), [m.replace("_", " ") for m in metrics])
        plt.xlabel("Relative Difference")
        plt.title("QA Cross-Validation: Metric Stability Comparison")
        plt.axvline(
            x=0.2, color="orange", linestyle="--", alpha=0.7, label="Good threshold"
        )
        plt.axvline(
            x=0.5, color="red", linestyle="--", alpha=0.7, label="Poor threshold"
        )
        plt.legend()
        plt.tight_layout()

        plot_file = output_dir / "qa_stability_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f" Visualization saved: {plot_file}")

    except ImportError:
        logging.warning("  Matplotlib not available - skipping visualization")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare QA results between two random subsets"
    )
    parser.add_argument("results_dir_a", help="First QA results directory")
    parser.add_argument("results_dir_b", help="Second QA results directory")
    parser.add_argument(
        "--output",
        default="qa_comparison",
        help="Output directory for comparison report",
    )
    parser.add_argument("--set-a-name", default="Set A", help="Name for first dataset")
    parser.add_argument("--set-b-name", default="Set B", help="Name for second dataset")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: validate inputs and show planned outputs without writing files",
    )
    # Print help when no args provided
    import sys

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()
    setup_logging()
    if args.dry_run:
        print("[DRY-RUN] QA cross-validation preview")
        print(f"[DRY-RUN] Inputs: {args.results_dir_a}, {args.results_dir_b}")
        print(f"[DRY-RUN] Output dir: {args.output}")
        return 0

    logging.info(" QA Cross-Validation Analysis")
    logging.info("=" * 50)

    # Load QA results
    qa_data_a = load_qa_results(args.results_dir_a)
    qa_data_b = load_qa_results(args.results_dir_b)

    if qa_data_a is None or qa_data_b is None:
        logging.error(" Failed to load QA data")
        return 1

    # Compute QA metrics
    qa_metrics_a = compute_qa_metrics(qa_data_a)
    qa_metrics_b = compute_qa_metrics(qa_data_b)

    # Compare metrics
    comparison = compare_qa_metrics(
        qa_metrics_a, qa_metrics_b, args.set_a_name, args.set_b_name
    )

    # Generate report
    generate_comparison_report(comparison, args.output)

    logging.info(f"\n QA cross-validation completed!")
    logging.info(f" Report saved to: {args.output}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
