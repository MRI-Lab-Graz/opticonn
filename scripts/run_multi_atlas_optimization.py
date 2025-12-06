#!/usr/bin/env python3
"""
Multi-Atlas Optimization Runner
===============================

Runs Bayesian Optimization separately for each atlas defined in the configuration.
This allows finding the optimal parameters specific to each atlas, rather than
optimizing for the average performance across all atlases.

Usage:
    python scripts/run_multi_atlas_optimization.py --config configs/my_config.json \
           --data-dir data/samples --output-dir results/study_name --n-iterations 30
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import copy
import time


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization separately for each atlas"
    )
    parser.add_argument("--config", required=True, help="Base configuration file")
    parser.add_argument(
        "--data-dir", required=True, help="Data directory containing .fz/.fib.gz files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Base output directory for the study"
    )
    parser.add_argument(
        "--n-iterations", type=int, default=20, help="Number of iterations per atlas"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would happen without running"
    )

    args = parser.parse_args()

    # Load base configuration
    try:
        with open(args.config) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1

    atlases = config.get("atlases", [])
    if not atlases:
        print("Error: No 'atlases' list found in the configuration file.")
        return 1

    print(f"\n{'='*60}")
    print(f" MULTI-ATLAS OPTIMIZATION RUNNER")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Data:   {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Atlases to optimize ({len(atlases)}):")
    for i, atlas in enumerate(atlases, 1):
        print(f"  {i}. {atlas}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry run complete. Exiting.")
        return 0

    results = []
    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    # Create a directory for the temporary per-atlas configs
    config_dir = base_output / "configs"
    config_dir.mkdir(exist_ok=True)

    start_time = time.time()

    for idx, atlas in enumerate(atlases, 1):
        print(f"\n>>> Processing Atlas {idx}/{len(atlases)}: {atlas}")
        print(f"{'-'*60}")

        # Create single-atlas config
        atlas_config = copy.deepcopy(config)
        atlas_config["atlases"] = [atlas]

        # Save temporary config
        temp_config_path = config_dir / f"config_{atlas}.json"
        with open(temp_config_path, "w") as f:
            json.dump(atlas_config, f, indent=2)

        # Define output directory for this specific atlas
        atlas_output_dir = base_output / atlas

        # Check if already done
        result_file = atlas_output_dir / "bayesian_optimization_results.json"
        if result_file.exists():
            print(f"  [INFO] Results already exist for {atlas}. Skipping...")
            # Load existing result for summary
            try:
                with open(result_file) as f:
                    res = json.load(f)
                    best_score = res.get("best_qa_score", res.get("best_score", 0.0))
                    best_params = res.get("best_parameters", res.get("best_params", {}))
                    results.append(
                        {
                            "atlas": atlas,
                            "score": best_score,
                            "params": best_params,
                            "status": "success (cached)",
                        }
                    )
            except Exception as e:
                print(f"  [WARNING] Could not read existing results for {atlas}: {e}")
            continue

        # Construct command
        cmd = [
            sys.executable,
            "scripts/bayesian_optimizer.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(atlas_output_dir),
            "--config",
            str(temp_config_path),
            "--n-iterations",
            str(args.n_iterations),
            # Pass verbose to see progress, or remove for cleaner output
            "--verbose",
        ]

        try:
            # Run the optimizer
            subprocess.run(cmd, check=True)

            # Retrieve results
            result_file = atlas_output_dir / "bayesian_optimization_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    res = json.load(f)
                    # Extract best score and params
                    # Note: bayesian_optimizer.py structure might vary, let's be robust
                    best_score = res.get("best_qa_score", res.get("best_score", 0.0))
                    best_params = res.get("best_parameters", res.get("best_params", {}))

                    results.append(
                        {
                            "atlas": atlas,
                            "score": best_score,
                            "params": best_params,
                            "status": "success",
                        }
                    )
                    print(f"  [SUCCESS] {atlas} Best Score: {best_score:.4f}")
            else:
                print(f"  [WARNING] No results file generated for {atlas}")
                results.append({"atlas": atlas, "score": 0.0, "status": "no_results"})

        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Optimization failed for {atlas}")
            results.append(
                {"atlas": atlas, "score": 0.0, "status": "failed", "error": str(e)}
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial results...")
            break

    total_time = time.time() - start_time

    # --- Summary Report ---
    print(f"\n{'='*60}")
    print(f" FINAL OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/60:.1f} minutes")

    # Sort results by score descending
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    print(f"\n{'Rank':<4} | {'Atlas':<30} | {'Score':<10} | {'Status'}")
    print(f"{'-'*60}")

    for i, r in enumerate(results, 1):
        print(
            f"{i:<4} | {r['atlas']:<30} | {r.get('score', 0):.4f}     | {r.get('status', 'unknown')}"
        )

    # Save summary to JSON
    summary_path = base_output / "multi_atlas_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"timestamp": time.ctime(), "config_used": args.config, "results": results},
            f,
            indent=2,
        )

    print(f"\nFull summary saved to: {summary_path}")

    # Identify the winner
    if results and results[0]["score"] > 0:
        winner = results[0]
        print(f"\n🏆 WINNER: {winner['atlas']} (Score: {winner['score']:.4f})")
        print("Best Parameters:")
        print(json.dumps(winner["params"], indent=2))


if __name__ == "__main__":
    main()
