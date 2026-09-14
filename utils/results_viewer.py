#!/usr/bin/env python3
"""
OptiConn Results Viewer
======================

Utility to view and analyze OptiConn optimization results.
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Optional


def load_candidates(candidates_file: Path) -> Optional[list]:
    """Load top candidates from results file."""
    try:
        with open(candidates_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Could not load candidates: {e}")
        return None


def show_top_candidates(candidates_file: Path, limit: int = 10) -> None:
    """Display top parameter candidates."""

    candidates = load_candidates(candidates_file)
    if not candidates:
        return

    print(f"🏆 TOP {min(limit, len(candidates))} PARAMETER CANDIDATES")
    print("=" * 80)

    for i, candidate in enumerate(candidates[:limit], 1):
        score = candidate.get("average_score", candidate.get("score", "N/A"))
        atlas = candidate.get("atlas", "N/A")
        metric = candidate.get("connectivity_metric", "N/A")

        print(f"\n#{i}: Score {score}")
        print(f"   Atlas: {atlas}")
        print(f"   Metric: {metric}")

        # Show parameters if available
        params = candidate.get("parameters", {})
        if params:
            print("   Parameters:")
            for param, value in params.items():
                if param != "tracking_parameters":
                    print(f"     {param}: {value}")

            # Show tracking parameters separately if they exist
            tracking = params.get("tracking_parameters", {})
            if tracking:
                print("   Tracking:")
                for param, value in tracking.items():
                    print(f"     {param}: {value}")


def compare_candidates(candidates_file: Path, indices: list[int]) -> None:
    """Compare specific candidates side by side."""

    candidates = load_candidates(candidates_file)
    if not candidates:
        return

    # Validate indices
    valid_indices = []
    for idx in indices:
        if 1 <= idx <= len(candidates):
            valid_indices.append(idx - 1)  # Convert to 0-based
        else:
            print(f"⚠️  Index {idx} out of range (1-{len(candidates)})")

    if not valid_indices:
        return

    print(f"🔍 COMPARING CANDIDATES {[i+1 for i in valid_indices]}")
    print("=" * 80)

    # Collect all unique parameter keys
    all_params = set()
    for idx in valid_indices:
        candidate = candidates[idx]
        params = candidate.get("parameters", {})
        all_params.update(params.keys())

        # Include tracking parameters
        tracking = params.get("tracking_parameters", {})
        for key in tracking.keys():
            all_params.add(f"tracking.{key}")

    # Create comparison table
    comparison_data = []

    for param in sorted(all_params):
        row = {"Parameter": param}

        for idx in valid_indices:
            candidate = candidates[idx]
            params = candidate.get("parameters", {})

            if param.startswith("tracking."):
                tracking_param = param.replace("tracking.", "")
                value = params.get("tracking_parameters", {}).get(tracking_param, "N/A")
            else:
                value = params.get(param, "N/A")

            score = candidate.get("average_score", candidate.get("score", "N/A"))
            row[f"#{idx+1} (score={score})"] = value

        comparison_data.append(row)

    # Display as table
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))


def analyze_pareto_front(results_dir: Path) -> None:
    """Analyze the Pareto front results."""

    pareto_file = results_dir / "pareto_front.csv"
    if not pareto_file.exists():
        print(f"❌ Pareto front file not found: {pareto_file}")
        return

    try:
        df = pd.read_csv(pareto_file)
        print("📈 PARETO FRONT ANALYSIS")
        print("=" * 50)
        print(f"Total combinations tested: {len(df)}")

        if "is_pareto_optimal" in df.columns:
            pareto_optimal = df[df["is_pareto_optimal"] == True]
            print(f"Pareto optimal combinations: {len(pareto_optimal)}")

        # Score statistics
        score_cols = [col for col in df.columns if "score" in col.lower()]
        if score_cols:
            score_col = score_cols[0]
            print(f"\nScore distribution ({score_col}):")
            print(f"  Best: {df[score_col].max():.3f}")
            print(f"  Worst: {df[score_col].min():.3f}")
            print(f"  Mean: {df[score_col].mean():.3f}")
            print(f"  Std: {df[score_col].std():.3f}")

        # Parameter ranges
        param_cols = [
            col for col in df.columns if col.endswith("_range") or "threshold" in col
        ]
        if param_cols:
            print("\nParameter ranges tested:")
            for col in param_cols[:5]:  # Show first 5
                unique_vals = df[col].nunique()
                print(f"  {col}: {unique_vals} unique values")

    except Exception as e:
        print(f"❌ Error analyzing Pareto front: {e}")


def generate_summary_report(
    results_dir: Path, output_file: Optional[Path] = None
) -> None:
    """Generate a comprehensive summary report."""

    if output_file is None:
        output_file = results_dir / "summary_report.txt"

    report_lines = []
    report_lines.append("OPTICONN OPTIMIZATION SUMMARY REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {pd.Timestamp.now()}")
    report_lines.append(f"Results directory: {results_dir}")
    report_lines.append("")

    # Top candidates
    candidates_file = results_dir / "top3_candidates.json"
    if candidates_file.exists():
        candidates = load_candidates(candidates_file)
        if candidates:
            report_lines.append("TOP CANDIDATES:")
            for i, candidate in enumerate(candidates[:3], 1):
                score = candidate.get("average_score", candidate.get("score", "N/A"))
                atlas = candidate.get("atlas", "N/A")
                metric = candidate.get("connectivity_metric", "N/A")
                report_lines.append(f"  #{i}: {atlas} + {metric} (score: {score})")
            report_lines.append("")

    # Pareto analysis
    pareto_file = results_dir / "pareto_front.csv"
    if pareto_file.exists():
        try:
            df = pd.read_csv(pareto_file)
            report_lines.append("OPTIMIZATION STATISTICS:")
            report_lines.append(f"  Total combinations tested: {len(df)}")

            score_cols = [col for col in df.columns if "score" in col.lower()]
            if score_cols:
                score_col = score_cols[0]
                report_lines.append(f"  Best score: {df[score_col].max():.3f}")
                report_lines.append(
                    f"  Score range: {df[score_col].min():.3f} - {df[score_col].max():.3f}"
                )

            report_lines.append("")
        except Exception:
            pass

    # Write report
    try:
        with open(output_file, "w") as f:
            f.write("\n".join(report_lines))
        print(f"📄 Summary report written to: {output_file}")
    except Exception as e:
        print(f"❌ Could not write report: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="OptiConn Results Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show top 5 candidates
  python results_viewer.py candidates --results results/optimize/optimization_results --limit 5
  
  # Compare specific candidates
  python results_viewer.py compare --results results/optimize/optimization_results --indices 1 2 3
  
  # Analyze Pareto front
  python results_viewer.py pareto --results results/optimize/optimization_results
  
  # Generate summary report
  python results_viewer.py report --results results/optimize/optimization_results
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Show candidates
    candidates_parser = subparsers.add_parser("candidates", help="Show top candidates")
    candidates_parser.add_argument(
        "--results", "-r", type=Path, required=True, help="Results directory"
    )
    candidates_parser.add_argument(
        "--limit", "-l", type=int, default=10, help="Number of candidates to show"
    )

    # Compare candidates
    compare_parser = subparsers.add_parser(
        "compare", help="Compare specific candidates"
    )
    compare_parser.add_argument(
        "--results", "-r", type=Path, required=True, help="Results directory"
    )
    compare_parser.add_argument(
        "--indices",
        "-i",
        type=int,
        nargs="+",
        required=True,
        help="Candidate indices to compare (1-based)",
    )

    # Pareto analysis
    pareto_parser = subparsers.add_parser("pareto", help="Analyze Pareto front")
    pareto_parser.add_argument(
        "--results", "-r", type=Path, required=True, help="Results directory"
    )

    # Generate report
    report_parser = subparsers.add_parser("report", help="Generate summary report")
    report_parser.add_argument(
        "--results", "-r", type=Path, required=True, help="Results directory"
    )
    report_parser.add_argument(
        "--output", "-o", type=Path, help="Output file (default: summary_report.txt)"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if not args.results.exists():
        print(f"❌ Results directory not found: {args.results}")
        return 1

    if args.command == "candidates":
        candidates_file = args.results / "top3_candidates.json"
        show_top_candidates(candidates_file, args.limit)

    elif args.command == "compare":
        candidates_file = args.results / "top3_candidates.json"
        compare_candidates(candidates_file, args.indices)

    elif args.command == "pareto":
        analyze_pareto_front(args.results)

    elif args.command == "report":
        generate_summary_report(args.results, args.output)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
