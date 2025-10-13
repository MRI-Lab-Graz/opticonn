#!/usr/bin/env python3
"""
OptiConn Hub CLI
=================

Central CLI that orchestrates optimization, analysis, and pipeline steps.
This module replaces the root-level braingraph.py by living inside scripts/
and resolving paths relative to the repository root automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scripts.utils.runtime import configure_stdio, propagate_no_emoji


def repo_root() -> Path:
    """Return repository root directory (parent of scripts/)."""
    # This file lives at <repo>/scripts/opticonn_hub.py
    return Path(__file__).resolve().parent.parent


def _abs(path_like: str | os.PathLike | None) -> str | None:
    if not path_like:
        return None
    return str(Path(path_like).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OptiConn - Unbiased, modality-agnostic connectomics optimization & analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
3-Step Workflow:
  1. opticonn sweep -i /path/to/data -o studies/run1 --quick
     → Compute connectivity & metrics for parameter combinations across waves

  2. opticonn review -o studies/run1/sweep-<uuid>/optimize
     → Auto-select best candidate based on QA+consistency (or use --interactive for GUI)

  3. opticonn apply --data-dir /path/to/full/dataset --optimal-config studies/run1/sweep-<uuid>/optimize/selected_candidate.json --output-dir studies/run1
     → Apply selected parameters to full dataset

Advanced:
  opticonn pipeline --step all --data-dir /path/to/fz --output studies/run2 --config my_config.json
  opticonn review -o studies/run1/sweep-<uuid>/optimize --interactive  # Launch web GUI for manual selection
        """,
    )

    parser.add_argument("--version", action="version", version="OptiConn v2.0.0")
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (useful on limited terminals)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: print the command(s) that would be executed without running them",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # review
    p_review = subparsers.add_parser(
        "review",
        help="Review sweep results and select best candidate (use --interactive for GUI)",
    )
    p_review.add_argument(
        "-o", "--output-dir", required=True, help="Sweep output directory to review"
    )
    p_review.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive web dashboard for manual candidate selection",
    )
    p_review.add_argument("--port", type=int, default=8050, help="Port for Dash app")
    p_review.add_argument(
        "--prune-nonbest",
        action="store_true",
        help="Delete non-optimal combo outputs after selection to save disk space (recommended for large sweeps)",
    )
    p_review.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (Windows-safe)",
    )

    # sweep
    p_sweep = subparsers.add_parser(
        "sweep", help="Run parameter sweep using cross-validation"
    )
    p_sweep.add_argument(
        "-i",
        "--data-dir",
        required=True,
        help="Directory containing .fz or .fib.gz files for sweep",
    )
    p_sweep.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for sweep results"
    )
    p_sweep.add_argument(
        "--config",
        help="Optional master sweep config (rare). If you want to provide an extraction/sweep config, prefer --extraction-config",
    )
    p_sweep.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny demonstration sweep (uses configs/sweep_micro.json)",
    )
    p_sweep.add_argument(
        "--subjects",
        type=int,
        default=3,
        help="Number of subjects to use for validation sweep (default: 3)",
    )
    # Advanced/parallel tuning
    p_sweep.add_argument(
        "--max-parallel", type=int, help="Max combinations to run in parallel per wave"
    )
    p_sweep.add_argument(
        "--extraction-config",
        help="Override extraction config for auto-generated waves",
    )
    # Reports and selection
    p_sweep.add_argument(
        "--no-report",
        action="store_true",
        help="Skip quick quality and Pareto reports after sweep",
    )
    p_sweep.add_argument(
        "--auto-select",
        action="store_true",
        help='[DEPRECATED] Use "opticonn review" (auto-select is now default) or "opticonn review --interactive" for GUI',
    )
    p_sweep.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (Windows-safe)",
    )
    p_sweep.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip full setup validation before running sweep",
    )
    p_sweep.add_argument(
        "--verbose",
        action="store_true",
        help="Show DSI Studio commands and detailed progress for each combination",
    )

    # apply
    p_apply = subparsers.add_parser(
        "apply",
        help="Apply optimal parameters to full dataset",
        description="Apply the optimal tractography parameters (selected via review) to your complete dataset. "
        "Runs full connectivity extraction and analysis pipeline with chosen settings.",
    )
    p_apply.add_argument(
        "-i",
        "--data-dir",
        required=True,
        help="Directory containing full dataset (.fz or .fib.gz files)",
    )
    p_apply.add_argument(
        "--optimal-config",
        required=True,
        help="Path to selected_candidate.json from review step",
    )
    p_apply.add_argument(
        "-o",
        "--output-dir",
        default="analysis_results",
        help="Output directory for final analysis results (default: analysis_results)",
    )
    p_apply.add_argument(
        "--analysis-only",
        action="store_true",
        help="Run only analysis on existing extraction outputs (skip connectivity extraction step)",
    )
    p_apply.add_argument(
        "--candidate-index",
        type=int,
        default=1,
        help="[Advanced] If optimal-config contains multiple candidates, select by 1-based index (default: 1 = best)",
    )
    p_apply.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and DSI Studio commands",
    )
    p_apply.add_argument(
        "--quiet", action="store_true", help="Reduce console output (minimal logging)"
    )
    p_apply.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (Windows-safe)",
    )

    # Legacy compatibility (will be removed in future version)
    p_apply.add_argument(
        "--skip-extraction",
        action="store_true",
        dest="analysis_only",
        help="[DEPRECATED] Use --analysis-only instead",
    )

    # bayesian (NEW)
    p_bayesian = subparsers.add_parser(
        "bayesian",
        help="🚀 Bayesian optimization for parameter search (efficient, smart)",
        description="Use Bayesian optimization to find optimal tractography parameters "
        "efficiently. Much faster than grid search (20-50 evaluations vs hundreds)."
    )
    p_bayesian.add_argument(
        "-i", "--data-dir",
        required=True,
        help="Directory containing .fz or .fib.gz files"
    )
    p_bayesian.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for Bayesian optimization results"
    )
    p_bayesian.add_argument(
        "--config",
        required=True,
        help="Base configuration JSON file"
    )
    p_bayesian.add_argument(
        "--n-iterations",
        type=int,
        default=30,
        help="Number of Bayesian optimization iterations (default: 30)"
    )
    p_bayesian.add_argument(
        "--n-bootstrap",
        type=int,
        default=3,
        help="Number of bootstrap samples per evaluation (default: 3)"
    )
    p_bayesian.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed optimization progress"
    )
    p_bayesian.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output"
    )

    # sensitivity (NEW)
    p_sensitivity = subparsers.add_parser(
        "sensitivity",
        help="📊 Analyze parameter sensitivity (which params matter most)",
        description="Perform sensitivity analysis to identify which tractography "
        "parameters have the most impact on network quality scores."
    )
    p_sensitivity.add_argument(
        "-i", "--data-dir",
        required=True,
        help="Directory containing .fz or .fib.gz files"
    )
    p_sensitivity.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for sensitivity analysis results"
    )
    p_sensitivity.add_argument(
        "--config",
        required=True,
        help="Baseline configuration JSON file"
    )
    p_sensitivity.add_argument(
        "--parameters",
        nargs='+',
        help="Specific parameters to analyze (default: all)"
    )
    p_sensitivity.add_argument(
        "--perturbation",
        type=float,
        default=0.1,
        help="Perturbation factor as fraction of baseline (default: 0.1 = 10%%)"
    )
    p_sensitivity.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis progress"
    )
    p_sensitivity.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output"
    )

    # pipeline
    p_pipe = subparsers.add_parser(
        "pipeline", help="Advanced pipeline execution (steps 01–03)"
    )
    p_pipe.add_argument(
        "--step", default="all", choices=["01", "02", "03", "all", "analysis"]
    )
    p_pipe.add_argument("-i", "--input")
    p_pipe.add_argument("-o", "--output")
    p_pipe.add_argument("--config")
    p_pipe.add_argument("--data-dir")
    p_pipe.add_argument("--cross-validated-config")
    p_pipe.add_argument("--quiet", action="store_true")
    p_pipe.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (Windows-safe)",
    )

    args = parser.parse_args()

    # Print help when called without args
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    root = repo_root()
    scripts_dir = root / "scripts"
    no_emoji = configure_stdio(getattr(args, "no_emoji", False))

    import uuid
    import subprocess

    def validate_json_config(config_path):
        validator_script = str(scripts_dir / "json_validator.py")
        result = subprocess.run(
            [sys.executable, validator_script, config_path, "--suggest-fixes"],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("Config validation failed. Exiting.")
            sys.exit(1)

    if args.command == "review":
        if not args.interactive:
            # Auto-select best candidate based on QA + wave consistency (DEFAULT)
            import json
            import glob

            optimize_dir = Path(args.output_dir)
            pattern = str(optimize_dir / "**/03_selection/optimal_combinations.json")
            files = glob.glob(pattern, recursive=True)

            if not files:
                print(
                    "❌ No optimal_combinations.json files found in optimize directory"
                )
                return 1

            # Load all candidates from all waves, with their parameters
            all_candidates = []
            wave_params_map = {}  # Map wave_name -> parameters

            for file_path in files:
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            wave_dir = Path(file_path).parent.parent
                            wave_name = wave_dir.name

                            # Load tracking parameters for this wave
                            params_file = wave_dir / "selected_parameters.json"
                            if params_file.exists():
                                with open(params_file, "r") as pf:
                                    params_data = json.load(pf)
                                    config = params_data.get(
                                        "selected_config", params_data
                                    )
                                    wave_params_map[wave_name] = config

                            for item in data:
                                item["wave"] = wave_name
                            all_candidates.extend(data)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load {file_path}: {e}")

            if not all_candidates:
                print("❌ No candidates found in optimal_combinations files")
                return 1

            # Find best candidate: highest QA score among those present in all waves
            import pandas as pd

            df = pd.DataFrame(all_candidates)
            df["candidate_key"] = df["atlas"] + "_" + df["connectivity_metric"]
            wave_counts = df.groupby("candidate_key")["wave"].nunique().to_dict()
            df["waves_present"] = df["candidate_key"].map(wave_counts)

            total_waves = df["wave"].nunique()
            consistent = df[df["waves_present"] == total_waves].copy()

            if consistent.empty:
                print(
                    f"⚠️  No candidates appear in all {total_waves} waves. Selecting best overall QA score..."
                )
                best_idx = df["pure_qa_score"].idxmax()
                best = df.loc[best_idx]
            else:
                # Among consistent candidates, pick highest avg QA
                avg_qa = df.groupby("candidate_key")["pure_qa_score"].mean().to_dict()
                consistent["avg_qa"] = consistent["candidate_key"].map(avg_qa)
                best_idx = consistent["avg_qa"].idxmax()
                best = consistent.loc[best_idx]

            best_dict = best.to_dict()

            # Attach tracking parameters from the winning wave
            best_wave = best_dict.get("wave")
            if best_wave and best_wave in wave_params_map:
                params = wave_params_map[best_wave]
                # Extract sweep_meta.choice for the parameters used
                sweep_meta = params.get("sweep_meta", {})
                choice = sweep_meta.get("choice", {})

                # Get full tracking parameters from config
                full_tracking = params.get("tracking_parameters", {})
                # Merge: sweep choice overrides defaults
                merged_tracking = {**full_tracking, **choice}

                best_dict["tracking_parameters"] = merged_tracking
                best_dict["tract_count"] = choice.get(
                    "tract_count", params.get("tract_count")
                )
                best_dict["connectivity_threshold"] = choice.get(
                    "connectivity_threshold"
                )

            # Save selection
            out_path = optimize_dir / "selected_candidate.json"
            with open(out_path, "w") as f:
                json.dump(
                    [best_dict], f, indent=2
                )  # Wrap in list for apply compatibility

            print(f"🏆 Auto-selected best candidate:")
            print(f"   Atlas: {best_dict['atlas']}")
            print(f"   Metric: {best_dict['connectivity_metric']}")
            print(f"   QA Score: {best_dict.get('pure_qa_score', 'N/A')}")
            print(f"   Waves present: {int(best_dict['waves_present'])}/{total_waves}")
            tp = best_dict.get("tracking_parameters", {})
            print(
                f"   Key params: n_tracks={best_dict.get('tract_count')}, fa={tp.get('fa_threshold')}, min_len={tp.get('min_length')}"
            )
            print(f"✅ Saved to: {out_path}")

            # Optionally prune non-best combo outputs to save disk space
            if args.prune_nonbest:
                import shutil

                print(f"\n🧹 Pruning non-optimal combination outputs...")
                best_combo_key = (
                    f"{best_dict['atlas']}_{best_dict['connectivity_metric']}"
                )

                # Find all wave directories
                wave_dirs = [
                    d
                    for d in optimize_dir.iterdir()
                    if d.is_dir() and d.name.startswith("wave")
                ]
                pruned_count = 0

                for wave_dir in wave_dirs:
                    combos_dir = wave_dir / "01_combos"
                    if not combos_dir.exists():
                        continue

                    for combo_dir in combos_dir.iterdir():
                        if not combo_dir.is_dir() or not combo_dir.name.startswith(
                            "sweep_"
                        ):
                            continue

                        # Check if this is the winning combo
                        combo_name = combo_dir.name
                        # Extract atlas and metric from directory name (format: sweep_<atlas>_<metric>_<hash>)
                        parts = combo_name.replace("sweep_", "").split("_")
                        if len(parts) >= 2:
                            combo_key = f"{parts[0]}_{parts[1]}"
                            if combo_key != best_combo_key:
                                try:
                                    shutil.rmtree(combo_dir)
                                    pruned_count += 1
                                except Exception as e:
                                    print(f"⚠️  Could not remove {combo_dir.name}: {e}")

                print(f"✅ Pruned {pruned_count} non-optimal combination directories")
                print(f"💾 Disk space saved!")

            # Try to extract data_dir from wave config for helpful hint
            data_dir_hint = "<your_full_dataset_directory>"
            try:
                wave_configs = list(optimize_dir.glob("configs/wave*.json"))
                if wave_configs:
                    import json

                    with open(wave_configs[0], "r") as f:
                        wave_cfg = json.load(f)
                        # Get parent directory of the sweep subset
                        sweep_data_dir = wave_cfg.get("data_selection", {}).get(
                            "source_dir", ""
                        )
                        if sweep_data_dir:
                            data_dir_hint = sweep_data_dir
            except Exception:
                pass

            print(
                f"\n👉 Next: opticonn apply --data-dir {data_dir_hint} --optimal-config {out_path} --output-dir <output_directory>"
            )
            return 0
        else:
            # Launch Dash app for interactive review (OPT-IN with --interactive)
            print("=" * 70)
            print("🎯 OPTICONN INTERACTIVE REVIEW")
            print("=" * 70)
            print(f"\n📊 Opening interactive dashboard to review sweep results...")
            print(f"🌐 The dashboard will open at: http://localhost:{args.port}")
            print(f"\n📋 Instructions:")
            print(f"   1. Review the candidates in the interactive dashboard")
            print(f"   2. Compare QA scores, network metrics, and Pareto plots")
            print(f"   3. Select your preferred candidate using the UI")
            print(
                f"   4. The selection will be saved to: {args.output_dir}/selected_candidate.json"
            )
            print(f"\n💡 After selection, run:")
            print(
                f"   opticonn apply --data-dir <your_full_dataset> --optimal-config {args.output_dir}/selected_candidate.json"
            )
            print(f"\n🚀 Launching dashboard...")
            print("=" * 70)

            dash_app = str(repo_root() / "scripts" / "dash_app" / "app.py")
            cmd = [
                sys.executable,
                dash_app,
                "--output",
                args.output_dir,
                "--port",
                str(args.port),
            ]
            subprocess.run(cmd)
            return 0

    if args.command == "sweep":
        # Run full setup validation unless opted out
        if not getattr(args, "no_validation", False):
            validate_script = str(scripts_dir / "validate_setup.py")
            # Try to auto-detect config and input for validation
            config_path = (
                args.config
                or args.extraction_config
                or str(root / "configs" / "braingraph_default_config.json")
            )
            input_path = args.data_dir
            output_path = args.output_dir
            val_args = [
                sys.executable,
                validate_script,
                "--config",
                config_path,
                "--output-dir",
                output_path,
                "--test-input",
                input_path,
            ]
            result = subprocess.run(val_args, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("❌ Full setup validation failed. Exiting.")
                sys.exit(1)
        # Append UUID to output directory
        unique_id = str(uuid.uuid4())
        sweep_output_dir = f"{_abs(args.output_dir)}/sweep-{unique_id}"
        cmd = [
            sys.executable,
            str(scripts_dir / "cross_validation_bootstrap_optimizer.py"),
            "--data-dir",
            _abs(args.data_dir),
            "--output-dir",
            sweep_output_dir,
        ]

        # Decide how to interpret provided configuration flags
        chosen_extraction_cfg: str | None = None
        chosen_master_cfg: str | None = None
        if args.quick:
            # Quick demo should use the tiny micro sweep to avoid large grids
            chosen_extraction_cfg = str(root / "configs" / "sweep_micro.json")
        if args.extraction_config:
            chosen_extraction_cfg = _abs(args.extraction_config)
        if args.config:
            try:
                import json as _json

                _cfg_path = Path(args.config)
                cfg_txt = _cfg_path.read_text()
                cfg_json = _json.loads(cfg_txt)
                is_master = any(
                    k in cfg_json
                    for k in ("wave1_config", "wave2_config", "bootstrap_optimization")
                )
                is_extraction_like = any(
                    k in cfg_json
                    for k in ("atlases", "connectivity_values", "sweep_parameters")
                )
                if is_master and not is_extraction_like:
                    chosen_master_cfg = _abs(args.config)
                else:
                    chosen_extraction_cfg = _abs(args.config)
            except Exception:
                chosen_extraction_cfg = _abs(args.config)

        # Validate configs before running sweep
        if chosen_master_cfg:
            validate_json_config(chosen_master_cfg)
            cmd += ["--config", chosen_master_cfg]
        if chosen_extraction_cfg:
            validate_json_config(chosen_extraction_cfg)
            cmd += ["--extraction-config", chosen_extraction_cfg]

        if args.subjects:
            cmd += ["--subjects", str(int(args.subjects))]
        if args.max_parallel and int(args.max_parallel) > 1:
            cmd += ["--max-parallel", str(int(args.max_parallel))]
        if args.verbose:
            cmd += ["--verbose"]
        if no_emoji:
            cmd.append("--no-emoji")
        if chosen_extraction_cfg:
            print(f"🧪 Using extraction config: {chosen_extraction_cfg}")
        if chosen_master_cfg:
            print(f"📋 Using master optimizer config: {chosen_master_cfg}")
        print(f"🚀 Running: {' '.join(cmd)}")
        print(f"🆔 Sweep output directory: {sweep_output_dir}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Parameter sweep completed successfully!")
            print(f"📋 Results saved to: {sweep_output_dir}/optimize")

            if not getattr(args, "no_report", False):
                # Autodetect network measures directory for quick quality check
                import glob

                selection_dirs = glob.glob(
                    f"{sweep_output_dir}/optimize/*/03_selection"
                )
                if selection_dirs:
                    matrices_dir = selection_dirs[0]
                    print(f"🔎 Running quick quality check on: {matrices_dir}")
                    qqc_script = str(root / "scripts" / "quick_quality_check.py")
                    qqc_args = [sys.executable, qqc_script, matrices_dir]
                    qqc_result = subprocess.run(
                        qqc_args, capture_output=True, text=True
                    )
                    print(qqc_result.stdout)
                    if qqc_result.returncode != 0:
                        print("⚠️  Quick quality check reported issues!")
                else:
                    print(
                        "⚠️  Could not find network measures directory for quick quality check."
                    )

                # Always run Pareto report if any wave diagnostics exist
                opt_dir = Path(sweep_output_dir) / "optimize"
                optimization_results_dir = opt_dir / "optimization_results"
                optimization_results_dir.mkdir(parents=True, exist_ok=True)
                wave_dirs = []
                for child in opt_dir.iterdir():
                    if child.is_dir() and (child / "combo_diagnostics.csv").exists():
                        wave_dirs.append(str(child.resolve()))
                if wave_dirs:
                    pareto_cmd = [
                        sys.executable,
                        str(root / "scripts" / "pareto_view.py"),
                        *wave_dirs,
                        "-o",
                        str(optimization_results_dir),
                        "--plot",
                    ]
                    print(f"📈 Generating Pareto report: {' '.join(pareto_cmd)}")
                    try:
                        subprocess.run(pareto_cmd, check=True, env=env)
                        print(
                            f"✅ Pareto report written to: {optimization_results_dir}"
                        )
                    except subprocess.CalledProcessError as e:
                        print(
                            f"⚠️  Pareto report generation failed with error code {e.returncode}"
                        )
                    except Exception as e:
                        print(f"⚠️  Pareto report generation encountered an error: {e}")
                else:
                    print(
                        "ℹ️  No wave diagnostics found (combo_diagnostics.csv); skipping Pareto report"
                    )

            # Conditional aggregation based on --auto-select flag
            optimize_dir = Path(sweep_output_dir) / "optimize"
            if args.auto_select:
                print("\n⚠️  WARNING: --auto-select is DEPRECATED")
                print(
                    "   Recommended: Use 'opticonn review' (auto-select is now default) or 'opticonn review --interactive' for GUI"
                )
                print("   Continuing with legacy mode...\n")
                print("🤖 Auto-selecting top candidates (legacy mode)...")
                try:
                    import subprocess

                    optimization_results_dir = optimize_dir / "optimization_results"
                    optimization_results_dir.mkdir(parents=True, exist_ok=True)
                    wave1_dir = optimize_dir / "bootstrap_qa_wave_1"
                    wave2_dir = optimize_dir / "bootstrap_qa_wave_2"
                    cmd_agg = [
                        sys.executable,
                        str(root / "scripts" / "aggregate_wave_candidates.py"),
                        str(optimization_results_dir),
                        str(wave1_dir),
                        str(wave2_dir),
                    ]
                    subprocess.run(cmd_agg, check=True)
                    top3 = optimization_results_dir / "top3_candidates.json"
                    print(f"✅ Auto-selected top 3 candidates: {top3}")
                    print(
                        f"👉 Next: opticonn apply -i {args.data_dir} --optimal-config {top3} -o {sweep_output_dir}"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to auto-aggregate candidates: {e}")
            else:
                print("\n" + "=" * 60)
                print("✅ SWEEP COMPLETE - Ready for Review")
                print("=" * 60)
                print(f"📊 Results: {optimize_dir}")
                print(f"\n👉 Next Step: Review results and select optimal parameters")
                print(f"   opticonn review -o {optimize_dir}")
                print(f"   (This will auto-select the best candidate. Add --interactive for GUI)")
                print(f"\n   Then apply selected parameters to full dataset:")
                print(
                    f"   opticonn apply -i {args.data_dir} --optimal-config {optimize_dir}/selected_candidate.json -o {sweep_output_dir}"
                )

            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Sweep failed with error code {e.returncode}")
            return e.returncode

    if args.command == "apply":
        # Determine if optimal-config is list (optimal_combinations.json) or dict
        import json

        cfg_path = Path(args.optimal_config)
        try:
            cfg_json = json.loads(Path(cfg_path).read_text())
        except Exception:
            cfg_json = None

        out_selected = Path(args.output_dir) / "selected"
        if isinstance(cfg_json, list):
            # Rank choices and pick candidate
            def score(obj):
                for k in ("average_score", "score", "pure_qa_score", "quality_score"):
                    v = obj.get(k)
                    if isinstance(v, (int, float)):
                        return float(v)
                pw = obj.get("per_wave")
                if isinstance(pw, list):
                    vals = [
                        w.get("score")
                        for w in pw
                        if isinstance(w, dict)
                        and isinstance(w.get("score"), (int, float))
                    ]
                    if vals:
                        return float(sum(vals) / len(vals))
                return 0.0

            ranked = sorted(cfg_json, key=score, reverse=True)
            idx = max(1, min(args.candidate_index, len(ranked))) - 1
            chosen = ranked[idx]

            # Resolve DSI Studio command
            dsi_cmd = os.environ.get("DSI_STUDIO_CMD")
            if (
                not dsi_cmd
                and (root / "configs" / "braingraph_default_config.json").exists()
            ):
                try:
                    default_cfg = json.loads(
                        (
                            root / "configs" / "braingraph_default_config.json"
                        ).read_text()
                    )
                    dsi_cmd = default_cfg.get("dsi_studio_cmd")
                except Exception:
                    dsi_cmd = None
            if not dsi_cmd:
                dsi_cmd = (
                    "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio"
                    if sys.platform == "darwin"
                    else "dsi_studio"
                )

            # Tentatively include parameter hints if present on the chosen candidate
            chosen_params = (
                chosen.get("parameters") if isinstance(chosen, dict) else None
            )
            extraction_cfg = {
                "description": "Extraction from selection (optimal_combinations.json)",
                "atlases": [chosen["atlas"]],
                "connectivity_values": [chosen["connectivity_metric"]],
                "dsi_studio_cmd": dsi_cmd,
            }
            # Merge selected parameter snapshot (non-destructive; downstream config loader tolerates missing fields)
            try:
                if isinstance(chosen_params, dict):
                    if "tract_count" in chosen_params:
                        extraction_cfg["tract_count"] = chosen_params["tract_count"]
                    tp = chosen_params.get("tracking_parameters") or {}
                    if tp:
                        extraction_cfg.setdefault("tracking_parameters", {})
                        for k in (
                            "fa_threshold",
                            "turning_angle",
                            "step_size",
                            "smoothing",
                            "min_length",
                            "max_length",
                            "track_voxel_ratio",
                            "dt_threshold",
                        ):
                            if tp.get(k) is not None:
                                extraction_cfg["tracking_parameters"][k] = tp.get(k)
                    ct = chosen_params.get("connectivity_threshold")
                    if ct is not None:
                        extraction_cfg.setdefault("connectivity_options", {})
                        extraction_cfg["connectivity_options"][
                            "connectivity_threshold"
                        ] = ct
            except Exception:
                pass
            out_selected.mkdir(parents=True, exist_ok=True)
            extraction_cfg_path = out_selected / "extraction_from_selection.json"
            extraction_cfg_path.write_text(json.dumps(extraction_cfg, indent=2))
            # Persist a selected_parameters.json for downstream Step 03 reporting
            try:
                (out_selected / "selected_parameters.json").write_text(
                    json.dumps({"selected_config": extraction_cfg}, indent=2)
                )
            except Exception:
                pass

            cmd = [
                sys.executable,
                str(root / "scripts" / "run_pipeline.py"),
                "--data-dir",
                _abs(args.data_dir),
                "--output",
                str(out_selected),
                "--extraction-config",
                str(extraction_cfg_path),
                "--step",
                "analysis" if args.analysis_only else "all",
            ]
            if args.verbose:
                print(f"🔍 Running with extraction config: {extraction_cfg_path}")
                print(
                    f"📊 Selected candidate: {chosen.get('atlas')} + {chosen.get('connectivity_metric')}"
                )
                cmd.append("--verbose")
            if args.quiet:
                cmd.append("--quiet")
        else:
            # Treat as cross-validated dict config
            cmd = [
                sys.executable,
                str(root / "scripts" / "run_pipeline.py"),
                "--cross-validated-config",
                _abs(args.optimal_config),
                "--data-dir",
                _abs(args.data_dir),
                "--output",
                str(out_selected),
                "--step",
                "analysis" if args.analysis_only else "all",
            ]
            if args.verbose:
                print(f"🔍 Running with cross-validated config: {args.optimal_config}")
                cmd.append("--verbose")
            if args.quiet:
                cmd.append("--quiet")

        if no_emoji:
            cmd.append("--no-emoji")

        # Validate config before running analysis/apply
        if isinstance(cfg_json, list):
            validate_json_config(str(extraction_cfg_path))
        else:
            validate_json_config(_abs(args.optimal_config))

        print(f"🚀 Running: {' '.join(cmd)}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Complete analysis finished successfully!")
            print(f"📋 Results available in: {out_selected}")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis failed with error code {e.returncode}")
            return e.returncode

    if args.command == "pipeline":
        cmd = [sys.executable, str(root / "scripts" / "run_pipeline.py")]
        config_path = None
        if args.step:
            cmd += ["--step", args.step]
        if args.input:
            cmd += ["--input", _abs(args.input)]
        if args.output:
            cmd += ["--output", _abs(args.output)]
        if args.config:
            config_path = _abs(args.config)
            cmd += ["--extraction-config", config_path]
        else:
            config_path = str(root / "configs" / "braingraph_default_config.json")
            cmd += ["--extraction-config", config_path]
        if args.data_dir:
            cmd += ["--data-dir", _abs(args.data_dir)]
        if args.cross_validated_config:
            cmd += ["--cross-validated-config", _abs(args.cross_validated_config)]
        if args.quiet:
            cmd.append("--quiet")
        if no_emoji:
            cmd.append("--no-emoji")

        # Validate config before running pipeline
        if config_path:
            validate_json_config(config_path)

        print(f"🚀 Running: {' '.join(cmd)}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Pipeline execution completed!")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline failed with error code {e.returncode}")
            return e.returncode

    if args.command == "bayesian":
        # Run Bayesian optimization
        cmd = [
            sys.executable,
            str(root / "scripts" / "bayesian_optimizer.py"),
            "-i", _abs(args.data_dir),
            "-o", _abs(args.output_dir),
            "--config", _abs(args.config),
            "--n-iterations", str(args.n_iterations),
            "--n-bootstrap", str(args.n_bootstrap),
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.no_emoji:
            cmd.append("--no-emoji")

        print(f"🚀 Starting Bayesian optimization...")
        print(f"   Data: {args.data_dir}")
        print(f"   Output: {args.output_dir}")
        print(f"   Iterations: {args.n_iterations}")
        
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Bayesian optimization completed!")
            print(f"\n📊 Results available in: {args.output_dir}")
            print(f"\n👉 Next: Apply the best parameters with 'opticonn apply'")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Bayesian optimization failed with error code {e.returncode}")
            return e.returncode

    if args.command == "sensitivity":
        # Run sensitivity analysis
        cmd = [
            sys.executable,
            str(root / "scripts" / "sensitivity_analyzer.py"),
            "-i", _abs(args.data_dir),
            "-o", _abs(args.output_dir),
            "--config", _abs(args.config),
            "--perturbation", str(args.perturbation),
        ]
        if args.parameters:
            cmd.extend(["--parameters"] + args.parameters)
        if args.verbose:
            cmd.append("--verbose")
        if args.no_emoji:
            cmd.append("--no-emoji")

        print(f"🚀 Starting sensitivity analysis...")
        print(f"   Data: {args.data_dir}")
        print(f"   Output: {args.output_dir}")
        if args.parameters:
            print(f"   Parameters: {', '.join(args.parameters)}")
        else:
            print(f"   Parameters: All")
        
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print("✅ Sensitivity analysis completed!")
            print(f"\n📊 Results available in: {args.output_dir}")
            print(f"   - sensitivity_analysis_results.json")
            print(f"   - sensitivity_analysis_plot.png")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"❌ Sensitivity analysis failed with error code {e.returncode}")
            return e.returncode

    print("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
