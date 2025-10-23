#!/usr/bin/env python3
# Supports: --dry-run (prints intended actions without running)
# When run without arguments the script prints help: parser.print_help()
"""
Parameter Uniqueness Verification
=================================

Optional utility (Extras): This script is not required for the core OptiConn
pipeline (Steps 01–03). It verifies that different parameter combinations
actually produce different connectivity results. Useful during development and
parameter sweep design to ensure uniqueness.

Author: Braingraph Pipeline Team
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from pathlib import Path
import argparse
import logging
from collections import defaultdict
import hashlib


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def compute_matrix_hash(matrix_file):
    """Compute hash of connectivity matrix to detect duplicates."""
    try:
        df = pd.read_csv(matrix_file)
        # Convert to numpy array and compute hash
        matrix_data = df.values.tobytes()
        return hashlib.md5(matrix_data).hexdigest()
    except Exception as e:
        logging.warning(f"Could not hash {matrix_file}: {e}")
        return None


def compute_matrix_stats(matrix_file):
    """Compute basic statistics of connectivity matrix."""
    try:
        df = pd.read_csv(matrix_file)
        matrix = df.values

        # Remove diagonal (self-connections)
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        off_diagonal = matrix[mask]

        stats = {
            "mean": np.mean(off_diagonal),
            "std": np.std(off_diagonal),
            "min": np.min(off_diagonal),
            "max": np.max(off_diagonal),
            "zeros": np.sum(off_diagonal == 0),
            "nonzeros": np.sum(off_diagonal != 0),
            "sparsity": np.sum(off_diagonal == 0) / len(off_diagonal),
        }
        return stats
    except Exception as e:
        logging.warning(f"Could not compute stats for {matrix_file}: {e}")
        return None


def analyze_parameter_uniqueness(matrices_dir):
    """Analyze if different parameters produce unique results."""
    logging.info("Starting parameter uniqueness analysis...")

    # Find all connectivity matrices
    csv_files = glob.glob(f"{matrices_dir}/**/by_atlas/**/*.csv", recursive=True)
    csv_files = [f for f in csv_files if not f.endswith("_network_measures.csv")]

    logging.info(f"Found {len(csv_files)} connectivity matrices")

    # Group by subject and atlas
    subject_atlas_groups = defaultdict(list)

    for csv_file in csv_files:
        path_parts = Path(csv_file).parts

        # Extract subject ID
        subject = None
        for part in path_parts:
            if part.startswith("sub-"):
                subject = part.split(".")[0]
                break

        # Extract atlas
        atlas = None
        filename = Path(csv_file).name
        if "FreeSurferDKT_Cortical" in filename:
            atlas = "FreeSurferDKT_Cortical"
        elif "FreeSurferDKT_Subcortical" in filename:
            atlas = "FreeSurferDKT_Subcortical"
        elif "HCP-MMP" in filename:
            atlas = "HCP-MMP"
        elif "AAL3" in filename:
            atlas = "AAL3"

        # Extract metric
        metric = None
        if ".count." in filename:
            metric = "count"
        elif ".fa." in filename:
            metric = "fa"
        elif ".qa." in filename:
            metric = "qa"
        elif ".ncount2." in filename:
            metric = "ncount2"

        if subject and atlas and metric:
            key = f"{subject}_{atlas}_{metric}"
            subject_atlas_groups[key].append(csv_file)

    logging.info(
        f"Grouped into {len(subject_atlas_groups)} subject/atlas/metric combinations"
    )

    # Analyze uniqueness within each group
    uniqueness_results = []
    duplicate_matrices = []

    for group_key, matrix_files in subject_atlas_groups.items():
        if len(matrix_files) < 2:
            continue

        logging.info(f"Analyzing group: {group_key} ({len(matrix_files)} matrices)")

        # Compute hashes and stats for all matrices in this group
        group_data = []
        for matrix_file in matrix_files:
            matrix_hash = compute_matrix_hash(matrix_file)
            matrix_stats = compute_matrix_stats(matrix_file)

            if matrix_hash and matrix_stats:
                group_data.append(
                    {"file": matrix_file, "hash": matrix_hash, "stats": matrix_stats}
                )

        # Check for duplicate hashes (identical matrices)
        hashes = [d["hash"] for d in group_data]
        unique_hashes = set(hashes)

        if len(unique_hashes) < len(hashes):
            duplicate_count = len(hashes) - len(unique_hashes)
            logging.warning(
                f"Group {group_key}: Found {duplicate_count} duplicate matrices!"
            )

            # Find which files are duplicates
            hash_to_files = defaultdict(list)
            for d in group_data:
                hash_to_files[d["hash"]].append(d["file"])

            for hash_val, files in hash_to_files.items():
                if len(files) > 1:
                    duplicate_matrices.append(
                        {"group": group_key, "hash": hash_val, "files": files}
                    )

        # Compute statistical diversity within group
        if len(group_data) > 1:
            means = [d["stats"]["mean"] for d in group_data]
            sparsities = [d["stats"]["sparsity"] for d in group_data]

            uniqueness_results.append(
                {
                    "group": group_key,
                    "n_matrices": len(group_data),
                    "unique_hashes": len(unique_hashes),
                    "mean_range": max(means) - min(means),
                    "mean_std": np.std(means),
                    "sparsity_range": max(sparsities) - min(sparsities),
                    "sparsity_std": np.std(sparsities),
                    "diversity_score": np.std(means) + np.std(sparsities),
                }
            )

    # Generate report
    results_df = pd.DataFrame(uniqueness_results)

    logging.info("\n" + "=" * 60)
    logging.info("PARAMETER UNIQUENESS ANALYSIS RESULTS")
    logging.info("=" * 60)

    if len(duplicate_matrices) > 0:
        logging.error(
            f" FOUND {len(duplicate_matrices)} GROUPS WITH DUPLICATE MATRICES!"
        )
        for dup in duplicate_matrices[:5]:  # Show first 5
            logging.error(f"Group: {dup['group']}")
            logging.error(
                f"Duplicate files: {dup['files'][:2]}..."
            )  # Show first 2 files
    else:
        logging.info(
            " NO DUPLICATE MATRICES FOUND - All parameter combinations produce unique results!"
        )

    if not results_df.empty:
        logging.info(f"\nDiversity Statistics (n={len(results_df)} groups):")
        logging.info(
            f"Mean connectivity range: {results_df['mean_range'].mean():.6f} ± {results_df['mean_range'].std():.6f}"
        )
        logging.info(
            f"Sparsity range: {results_df['sparsity_range'].mean():.3f} ± {results_df['sparsity_range'].std():.3f}"
        )
        logging.info(
            f"Overall diversity score: {results_df['diversity_score'].mean():.6f}"
        )

        # Flag groups with suspiciously low diversity
        low_diversity = results_df[
            results_df["diversity_score"] < results_df["diversity_score"].quantile(0.1)
        ]
        if not low_diversity.empty:
            logging.warning(f"\n  Groups with low parameter diversity (bottom 10%):")
            for _, row in low_diversity.iterrows():
                logging.warning(
                    f"  {row['group']}: diversity_score = {row['diversity_score']:.6f}"
                )

    return {
        "duplicate_matrices": duplicate_matrices,
        "uniqueness_results": results_df,
        "summary": {
            "total_groups": len(subject_atlas_groups),
            "groups_analyzed": len(uniqueness_results),
            "duplicate_groups": len(duplicate_matrices),
            "mean_diversity": (
                results_df["diversity_score"].mean() if not results_df.empty else 0
            ),
        },
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify parameter sweep uniqueness")
    parser.add_argument(
        "matrices_dir", help="Directory containing connectivity matrices"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for results",
        default="uniqueness_check_results",
    )

    args = parser.parse_args()
    setup_logging()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run analysis
    results = analyze_parameter_uniqueness(args.matrices_dir)

    # Save detailed results
    if (
        results["uniqueness_results"] is not None
        and not results["uniqueness_results"].empty
    ):
        results_file = os.path.join(args.output, "parameter_uniqueness_results.csv")
        results["uniqueness_results"].to_csv(results_file, index=False)
        logging.info(f"Detailed results saved to: {results_file}")

    # Save duplicate report
    if results["duplicate_matrices"]:
        duplicates_file = os.path.join(args.output, "duplicate_matrices.json")
        import json

        with open(duplicates_file, "w") as f:
            json.dump(results["duplicate_matrices"], f, indent=2)
        logging.info(f"Duplicate matrices report saved to: {duplicates_file}")

    # Summary
    logging.info(f"\n SUMMARY:")
    logging.info(f"Total groups: {results['summary']['total_groups']}")
    logging.info(f"Duplicate groups: {results['summary']['duplicate_groups']}")
    logging.info(f"Parameter diversity: {results['summary']['mean_diversity']:.6f}")

    return 0 if results["summary"]["duplicate_groups"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
