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
        epilog=(
            "3-Step Workflow:\n"
            "  1. opticonn tune-bayes -i /path/to/data -o studies/run1 --config configs/braingraph_default_config.json\n"
            "      → Discover optimal parameters efficiently with Bayesian optimization (recommended)\n\n"
            "  2. opticonn select -i studies/run1 --modality qa\n"
            "      → Confirm the best candidate and prep config for application\n\n"
            "  3. opticonn apply --data-dir /path/to/full/dataset --optimal-config studies/run1/qa/bayesian_optimization_results.json --output-dir studies/run1\n"
            "      → Apply selected parameters to full dataset\n\n"
            "Alternative baseline:\n"
            "  opticonn tune-grid -i /path/to/pilot_data -o studies/run1 --quick\n"
            "  opticonn select -i studies/run1/sweep-<uuid>/optimize\n\n"
            "Advanced:\n"
            "  opticonn pipeline --step all --data-dir /path/to/fz --output studies/run2 --config my_config.json\n"
        ),
    )

    parser.add_argument("--version", action="version", version="OptiConn v2.0.0")
    parser.add_argument(
        "--backend",
        choices=["dsi", "mrtrix"],
        default="mrtrix",
        help="Tractography backend to use (default: mrtrix). Use 'dsi' for legacy DSI Studio workflows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: print the command(s) that would be executed without running them",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # select
    p_select = subparsers.add_parser(
        "select",
        help="Select the best candidate from Bayesian or grid tuning outputs",
    )
    p_select.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="Path to tuning output: sweep optimize directory or Bayesian results JSON file",
    )
    p_select.add_argument(
        "--modality",
        default=None,
        help=(
            "When --input-path points to a per-modality tune-bayes output directory, "
            "select which modality to use (e.g., qa, fa)."
        ),
    )
    p_select.add_argument(
        "--prune-nonbest",
        action="store_true",
        help="For grid outputs, delete non-optimal combo results after selection to save disk space",
    )

    # mrtrix-discover
    p_mrtrix_discover = subparsers.add_parser(
        "mrtrix-discover",
        help="[MRtrix only] Discover tracking-ready bundle from QSIRecon outputs",
    )
    p_mrtrix_discover.add_argument(
        "--qsirecon-dir",
        help="Path to QSIRecon derivatives directory",
    )
    p_mrtrix_discover.add_argument(
        "--qsiprep-dir",
        help="Path to QSIPrep derivatives directory (optional, used for masks)",
    )
    p_mrtrix_discover.add_argument(
        "--derivatives-dir",
        help="Path to a root derivatives directory (auto-finds qsirecon/qsiprep)",
    )
    p_mrtrix_discover.add_argument(
        "-o",
        "--output",
        help="Path to write the discovered bundle JSON config",
    )
    p_mrtrix_discover.add_argument(
        "--subject",
        help="Subject ID to discover (e.g., sub-01). If omitted, finds first available.",
    )
    p_mrtrix_discover.add_argument(
        "--session",
        help="Session ID to discover (e.g., ses-01).",
    )
    p_mrtrix_discover.add_argument(
        "--atlas",
        required=True,
        help="Atlas name to discover (e.g., desikan, Brainnetome246Ext).",
    )

    # tune-grid
    p_tune_grid = subparsers.add_parser(
        "tune-grid", help="Run grid/random tuning with cross-validation"
    )
    p_tune_grid.add_argument(
        "-i",
        "--data-dir",
        required=True,
        help="Directory containing .fz or .fib.gz files for tuning",
    )
    p_tune_grid.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for tuning results",
    )
    p_tune_grid.add_argument(
        "--config",
        help="Optional master sweep config (rare). If you want to provide an extraction/tuning config, prefer --extraction-config",
    )
    p_tune_grid.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny demonstration tuning run (uses configs/sweep_micro.json)",
    )
    p_tune_grid.add_argument(
        "--subjects",
        type=int,
        default=3,
        help="Number of subjects to use for validation (default: 3)",
    )
    p_tune_grid.add_argument(
        "--subject",
        help="[MRtrix only] Subject ID to optimize (e.g., sub-01)",
    )
    # Advanced/parallel tuning
    p_tune_grid.add_argument(
        "--max-parallel",
        type=int,
        help="Max combinations to run in parallel per wave",
    )
    p_tune_grid.add_argument(
        "--extraction-config",
        help="Override extraction config for auto-generated waves",
    )
    # Reports and selection
    p_tune_grid.add_argument(
        "--no-report",
        action="store_true",
        help="Skip quick quality and Pareto reports after tuning",
    )
    p_tune_grid.add_argument(
        "--auto-select",
        action="store_true",
        help='[DEPRECATED] Auto-selection now happens via "opticonn select"',
    )
    p_tune_grid.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip full setup validation before running",
    )
    p_tune_grid.add_argument(
        "--verbose",
        action="store_true",
        help="Show DSI Studio commands and detailed progress for each combination",
    )

    # apply
    p_apply = subparsers.add_parser(
        "apply",
        help="Apply optimal parameters to full dataset",
        description="Apply the optimal tractography parameters (selected via opticonn select) to your complete dataset. "
        "Runs full connectivity extraction and analysis pipeline with chosen settings.",
    )
    p_apply.add_argument(
        "-i",
        "--data-dir",
        required=True,
        help="Directory containing full dataset (.fz/.fib.gz for DSI, or derivatives for MRtrix)",
    )
    p_apply.add_argument(
        "--optimal-config",
        required=True,
        help="Path to selected_candidate.json from selection step",
    )
    p_apply.add_argument(
        "-o",
        "--output-dir",
        default="analysis_results",
        help="Output directory for final analysis results (default: analysis_results)",
    )
    p_apply.add_argument(
        "--backend",
        choices=["dsi", "mrtrix"],
        default=None,
        help="Override backend (default: auto-detect from config)",
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
        "--subject",
        help="[MRtrix only] Subject ID to process (e.g., sub-01)",
    )
    p_apply.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and DSI Studio commands",
    )
    p_apply.add_argument(
        "--quiet", action="store_true", help="Reduce console output (minimal logging)"
    )
    # Legacy compatibility (will be removed in future version)
    p_apply.add_argument(
        "--skip-extraction",
        action="store_true",
        dest="analysis_only",
        help="[DEPRECATED] Use --analysis-only instead",
    )

    # tune-bayes (NEW)
    p_tune_bayes = subparsers.add_parser(
        "tune-bayes",
        help="Bayesian optimization for parameter search (efficient, smart)",
        description="Use Bayesian optimization to find optimal tractography parameters "
        "efficiently. Much faster than grid search (20-50 evaluations vs hundreds).",
    )
    p_tune_bayes.add_argument(
        "-i",
        "--data-dir",
        required=True,
        help="Directory containing .fz or .fib.gz files",
    )
    p_tune_bayes.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for Bayesian optimization results",
    )
    p_tune_bayes.add_argument(
        "--config", required=False, help="Base configuration JSON file (required for DSI backend)"
    )
    p_tune_bayes.add_argument(
        "--n-iterations",
        type=int,
        default=30,
        help="Number of Bayesian optimization iterations (default: 30)",
    )
    p_tune_bayes.add_argument(
        "--n-bootstrap",
        type=int,
        default=3,
        help="Number of bootstrap samples per evaluation (default: 3)",
    )
    p_tune_bayes.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers for evaluations (default: 1 = sequential). Use 2-4 for parallel execution.",
    )
    p_tune_bayes.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help=(
            "One or more connectivity modalities to optimize separately (e.g., qa fa). "
            "If omitted, defaults to a single modality inferred from the config (prefers 'qa' if present)."
        ),
    )
    p_tune_bayes.add_argument(
        "--sample-subjects",
        action="store_true",
        help="Sample different subject per iteration (faster, recommended). Default: use all subjects.",
    )
    p_tune_bayes.add_argument(
        "--subject",
        help="[MRtrix only] Subject ID to optimize (e.g., sub-01)",
    )
    p_tune_bayes.add_argument(
        "--verbose", action="store_true", help="Show detailed optimization progress"
    )
    # sensitivity (NEW)
    p_sensitivity = subparsers.add_parser(
        "sensitivity",
        help=" Analyze parameter sensitivity (which params matter most)",
        description="Perform sensitivity analysis to identify which tractography "
        "parameters have the most impact on network quality scores.",
    )
    p_sensitivity.add_argument(
        "-i",
        "--data-dir",
        required=True,
        help="Directory containing .fz or .fib.gz files",
    )
    p_sensitivity.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for sensitivity analysis results",
    )
    p_sensitivity.add_argument(
        "--config", required=True, help="Baseline configuration JSON file"
    )
    p_sensitivity.add_argument(
        "--parameters", nargs="+", help="Specific parameters to analyze (default: all)"
    )
    p_sensitivity.add_argument(
        "--perturbation",
        type=float,
        default=0.1,
        help="Perturbation factor as fraction of baseline (default: 0.1 = 10%%)",
    )
    p_sensitivity.add_argument(
        "--verbose", action="store_true", help="Show detailed analysis progress"
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
    p_pipe.add_argument(
        "--backend",
        choices=["dsi", "mrtrix"],
        default="mrtrix",
        help="Tractography backend to use (default: mrtrix)",
    )
    p_pipe.add_argument("--cross-validated-config")
    p_pipe.add_argument("--quiet", action="store_true")
    # Print help when called without args
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    root = repo_root()
    scripts_dir = root / "scripts"
    configure_stdio()

    import uuid
    import subprocess

    # Minimal ANSI coloring (auto-disabled when not appropriate)
    def _use_color() -> bool:
        if os.environ.get("NO_COLOR") is not None:
            return False
        if os.environ.get("TERM", "").lower() == "dumb":
            return False
        try:
            return sys.stdout.isatty()
        except Exception:
            return False

    _COLOR = _use_color()
    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _CYAN = "\033[36m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _MAGENTA = "\033[35m"
    _RED = "\033[31m"

    def _c(text: str, ansi: str) -> str:
        return f"{ansi}{text}{_RESET}" if _COLOR else text

    def _step(title: str) -> str:
        return _c(f"[{title}]", _BOLD + _CYAN)

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

    if args.command == "select":
        import json as _json

        input_path = Path(args.input_path)

        def _select_from_bayes_results(results_json: Path) -> int:
            print(f"{_step('SELECT')} Bayesian results: {results_json}")
            try:
                with open(results_json, "r") as f:
                    data = _json.load(f)

                best_params = data.get("best_parameters") or data.get("best_params")
                best_value = (
                    data.get("best_quality_score")
                    or data.get("best_qa_score")
                    or data.get("best_value")
                    or data.get("best_score")
                )

                if not best_params:
                    print(
                        _c(
                            "ERROR: No 'best_parameters' or 'best_params' key found in the JSON file.",
                            _RED,
                        )
                    )
                    return 1

                modality = data.get("target_modality")
                if modality:
                    print(f"{_c('Modality:', _BOLD)} {modality}")

                print(_c("\nBest Parameters:", _BOLD + _GREEN))
                for key, value in best_params.items():
                    print(f"   - {key}: {value}")

                if best_value is not None:
                    try:
                        print(f"\nBest Score: {float(best_value):.4f}")
                    except Exception:
                        print(f"\nBest Score: {best_value}")

                total = data.get("n_iterations")
                completed = data.get("completed_iterations")
                if total and completed:
                    print(f"\nProgress: {completed}/{total} iterations completed")

                print(_c("\nNext:", _BOLD + _YELLOW))
                print(
                    f"   opticonn apply --data-dir <your_full_dataset> --optimal-config {results_json.resolve()} -o <output_directory>"
                )
                return 0

            except Exception as e:
                print(_c(f"Error reading or parsing JSON file: {e}", _RED))
                return 1

        if input_path.is_file() and input_path.suffix == ".json":
            return _select_from_bayes_results(input_path)

        elif input_path.is_dir():
            # Per-modality tune-bayes output directory support.
            manifest_path = input_path / "bayesian_optimization_manifest.json"
            nested_results = list(
                input_path.glob("*/bayesian_optimization_results.json")
            )
            if manifest_path.exists() or nested_results:
                modalities: list[str] = []
                modality_to_results: dict[str, Path] = {}

                if manifest_path.exists():
                    try:
                        manifest = _json.loads(manifest_path.read_text())
                        for item in manifest.get("modalities") or []:
                            m = item.get("modality")
                            rf = item.get("results_file")
                            if isinstance(m, str) and m:
                                modalities.append(m)
                                if isinstance(rf, str) and rf:
                                    modality_to_results[m] = Path(rf)
                    except Exception:
                        modalities = []
                        modality_to_results = {}

                if not modalities:
                    for p in sorted(nested_results):
                        m = p.parent.name
                        modalities.append(m)
                        modality_to_results[m] = p

                modalities = list(dict.fromkeys(modalities))
                if not modalities:
                    print(_c("No modality results found in directory.", _RED))
                    return 1

                chosen = args.modality
                if chosen is None:
                    if len(modalities) == 1:
                        chosen = modalities[0]
                    else:
                        if sys.stdin.isatty() and sys.stdout.isatty():
                            print(f"{_step('SELECT')} Multiple modalities found:")
                            for idx, m in enumerate(modalities, start=1):
                                print(f"  {idx}) {m}")
                            raw = input("Pick modality (name or number): ").strip()
                            if raw.isdigit():
                                i = int(raw)
                                if 1 <= i <= len(modalities):
                                    chosen = modalities[i - 1]
                            elif raw in modalities:
                                chosen = raw

                        if chosen is None:
                            print(
                                _c(
                                    f"Multiple modalities available; re-run with --modality one of: {', '.join(modalities)}",
                                    _YELLOW,
                                )
                            )
                            return 2

                if chosen not in modality_to_results:
                    print(
                        _c(
                            f"Unknown modality '{chosen}'. Available: {', '.join(modalities)}",
                            _RED,
                        )
                    )
                    return 2

                return _select_from_bayes_results(modality_to_results[chosen])

            # Handle grid-tuning results directory (existing logic)
            # Auto-select best candidate based on QA + wave consistency (DEFAULT)
            import glob

            optimize_dir = input_path
            pattern = str(optimize_dir / "**/03_selection/optimal_combinations.json")
            files = glob.glob(pattern, recursive=True)

            if not files:
                print(" No optimal_combinations.json files found in optimize directory")
                return 1

            # Load all candidates from all waves, with their parameters
            all_candidates = []
            wave_params_map = {}  # Map wave_name -> parameters

            for file_path in files:
                try:
                    with open(file_path, "r") as f:
                        data = _json.load(f)
                        if isinstance(data, list):
                            wave_dir = Path(file_path).parent.parent
                            wave_name = wave_dir.name

                            # Load tracking parameters for this wave
                            params_file = wave_dir / "selected_parameters.json"
                            if params_file.exists():
                                with open(params_file, "r") as pf:
                                    params_data = _json.load(pf)
                                    config = params_data.get(
                                        "selected_config", params_data
                                    )
                                    wave_params_map[wave_name] = config

                            for item in data:
                                item["wave"] = wave_name
                            all_candidates.extend(data)
                except Exception as e:
                    print(f"  Warning: Could not load {file_path}: {e}")

            if not all_candidates:
                print(" No candidates found in optimal_combinations files")
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
                    f"  No candidates appear in all {total_waves} waves. Selecting best overall QA score..."
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
                _json.dump(
                    [best_dict], f, indent=2
                )  # Wrap in list for apply compatibility

            print(" Auto-selected best candidate:")
            print(f"   Atlas: {best_dict['atlas']}")
            print(f"   Metric: {best_dict['connectivity_metric']}")
            print(f"   QA Score: {best_dict.get('pure_qa_score', 'N/A')}")
            print(f"   Waves present: {int(best_dict['waves_present'])}/{total_waves}")
            tp = best_dict.get("tracking_parameters", {})
            print(
                f"   Key params: n_tracks={best_dict.get('tract_count')}, fa={tp.get('fa_threshold')}, min_len={tp.get('min_length')}"
            )
            print(f" Saved to: {out_path}")

            # Optionally prune non-best combo outputs to save disk space
            if args.prune_nonbest:
                import shutil

                print("\n Pruning non-optimal combination outputs...")
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
                                    print(f"  Could not remove {combo_dir.name}: {e}")

                print(f" Pruned {pruned_count} non-optimal combination directories")
                print(" Disk space saved!")

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
                f"\n Next: opticonn apply --data-dir {data_dir_hint} --optimal-config {out_path} --output-dir <output_directory>"
            )
            return 0
        else:
            print(f" Input path is not a valid file or directory: {input_path}")
            return 1

    if args.command == "mrtrix-discover":
        cmd = [
            sys.executable,
            str(root / "scripts" / "mrtrix_discover_bundle.py"),
        ]
        if args.qsirecon_dir:
            cmd += ["--qsirecon-dir", _abs(args.qsirecon_dir)]
        if args.qsiprep_dir:
            cmd += ["--qsiprep-dir", _abs(args.qsiprep_dir)]
        if args.derivatives_dir:
            cmd += ["--derivatives-dir", _abs(args.derivatives_dir)]
        if args.output:
            cmd += ["--out-config", _abs(args.output)]
        if args.subject:
            cmd += ["--subject", args.subject]
        if args.session:
            cmd += ["--session", args.session]
        if args.atlas:
            cmd += ["--atlas", args.atlas]

        print(f" Running MRtrix bundle discovery: {' '.join(cmd)}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            return 0
        except subprocess.CalledProcessError as e:
            print(f" Discovery failed with error code {e.returncode}")
            return e.returncode

    if args.command == "tune-grid":
        if args.backend == "mrtrix":
            # Dispatch to mrtrix_tune.py sweep
            cmd = [
                sys.executable,
                str(root / "scripts" / "mrtrix_tune.py"),
                "sweep",
                "--output-dir",
                _abs(args.output_dir),
            ]
            if args.data_dir:
                if Path(args.data_dir).is_file():
                    cmd += ["--config", _abs(args.data_dir)]
                else:
                    cmd += ["--derivatives-dir", _abs(args.data_dir)]
            if args.subject:
                cmd += ["--subject", args.subject]
            if args.max_parallel:
                cmd += ["--nthreads", str(args.max_parallel)]
            if args.verbose:
                cmd.append("--verbose")
            if getattr(args, "dry_run", False):
                cmd.append("--dry-run")

            print(f" Running MRtrix grid tuning: {' '.join(cmd)}")
            env = propagate_no_emoji()
            try:
                subprocess.run(cmd, check=True, env=env)
                return 0
            except subprocess.CalledProcessError as e:
                return e.returncode

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
                print(" Full setup validation failed. Exiting.")
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
        if chosen_extraction_cfg:
            print(f" Using extraction config: {chosen_extraction_cfg}")
        if chosen_master_cfg:
            print(f" Using master optimizer config: {chosen_master_cfg}")
        print(f" Running: {' '.join(cmd)}")
        print(f" Tuning output directory: {sweep_output_dir}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print(" Grid tuning completed successfully!")
            print(f" Results saved to: {sweep_output_dir}/optimize")

            if not getattr(args, "no_report", False):
                # Autodetect network measures directory for quick quality check
                import glob

                selection_dirs = glob.glob(
                    f"{sweep_output_dir}/optimize/*/03_selection"
                )
                if selection_dirs:
                    matrices_dir = selection_dirs[0]
                    print(f" Running quick quality check on: {matrices_dir}")
                    qqc_script = str(root / "scripts" / "quick_quality_check.py")
                    qqc_args = [sys.executable, qqc_script, matrices_dir]
                    qqc_result = subprocess.run(
                        qqc_args, capture_output=True, text=True
                    )
                    print(qqc_result.stdout)
                    if qqc_result.returncode != 0:
                        print("  Quick quality check reported issues!")
                else:
                    print(
                        "  Could not find network measures directory for quick quality check."
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
                    print(f" Generating Pareto report: {' '.join(pareto_cmd)}")
                    try:
                        subprocess.run(pareto_cmd, check=True, env=env)
                        print(f" Pareto report written to: {optimization_results_dir}")
                    except subprocess.CalledProcessError as e:
                        print(
                            f"  Pareto report generation failed with error code {e.returncode}"
                        )
                    except Exception as e:
                        print(f"  Pareto report generation encountered an error: {e}")
                else:
                    print(
                        "ℹ️  No wave diagnostics found (combo_diagnostics.csv); skipping Pareto report"
                    )

            # Conditional aggregation based on --auto-select flag
            optimize_dir = Path(sweep_output_dir) / "optimize"
            if args.auto_select:
                print("\n  WARNING: --auto-select is DEPRECATED")
                print(
                    "   Recommended: Use 'opticonn select' (auto-select is now default)"
                )
                print("   Continuing with legacy mode...\n")
                print(" Auto-selecting top candidates (legacy mode)...")
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
                    print(f" Auto-selected top 3 candidates: {top3}")
                    print(
                        f" Next: opticonn apply -i {args.data_dir} --optimal-config {top3} -o {sweep_output_dir}"
                    )
                except Exception as e:
                    print(f"  Failed to auto-aggregate candidates: {e}")
            else:
                print("\n" + "=" * 60)
                print(" GRID TUNING COMPLETE - Ready for Selection")
                print("=" * 60)
                print(f" Results: {optimize_dir}")
                print("\n Next Step: Select results and promote optimal parameters")
                print(f"   opticonn select -i {optimize_dir}")
                print("   (This will auto-select the best candidate)")
                print("\n   Then apply selected parameters to full dataset:")
                print(
                    f"   opticonn apply -i {args.data_dir} --optimal-config {optimize_dir}/selected_candidate.json -o {sweep_output_dir}"
                )

            return 0
        except subprocess.CalledProcessError as e:
            print(f" Grid tuning failed with error code {e.returncode}")
            return e.returncode

    if args.command == "apply":
        # Determine if optimal-config is list (optimal_combinations.json) or dict
        import json

        cfg_path = Path(args.optimal_config)
        try:
            cfg_json = json.loads(Path(cfg_path).read_text())
        except Exception:
            cfg_json = None

        # Auto-detect backend if not provided
        backend = args.backend
        if backend is None:
            if isinstance(cfg_json, dict) and cfg_json.get("run_metadata", {}).get("backend") == "mrtrix":
                backend = "mrtrix"
            elif isinstance(cfg_json, list) and len(cfg_json) > 0 and "mrtrix" in str(cfg_json[0].get("wave", "")):
                backend = "mrtrix"
            else:
                backend = "dsi"

        if backend == "mrtrix":
            cmd = [
                sys.executable,
                str(root / "scripts" / "mrtrix_tune.py"),
                "apply",
                "--output-dir", _abs(args.output_dir),
                "--optimal-config", _abs(args.optimal_config),
            ]
            if args.data_dir:
                if Path(args.data_dir).is_file():
                    cmd += ["--config", _abs(args.data_dir)]
                else:
                    cmd += ["--derivatives-dir", _abs(args.data_dir)]
            if args.subject:
                cmd += ["--subject", args.subject]
            if args.verbose:
                cmd.append("--verbose")
            if getattr(args, "dry_run", False):
                cmd.append("--dry-run")
            
            print(f" Running MRtrix application: {' '.join(cmd)}")
            env = propagate_no_emoji()
            try:
                subprocess.run(cmd, check=True, env=env)
                return 0
            except subprocess.CalledProcessError as e:
                return e.returncode

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
                print(f" Running with extraction config: {extraction_cfg_path}")
                print(
                    f" Selected candidate: {chosen.get('atlas')} + {chosen.get('connectivity_metric')}"
                )
                cmd.append("--verbose")
            if args.quiet:
                cmd.append("--quiet")
        else:
            # Treat as Bayesian optimization result, loading defaults and merging
            # optimal parameters on top.
            default_cfg_path = root / "configs" / "braingraph_default_config.json"
            if not default_cfg_path.exists():
                print(f" Default config not found at: {default_cfg_path}")
                return 1

            try:
                with open(cfg_path, "r") as f:
                    optimal_data = json.load(f)

                # If the provided JSON already looks like a full extraction config
                # (e.g., demo-generated apply_config_<modality>.json), use it as base.
                # Otherwise, fall back to the default config and only merge best_parameters.
                if isinstance(optimal_data, dict) and any(
                    k in optimal_data
                    for k in (
                        "atlases",
                        "connectivity_values",
                        "tracking_parameters",
                        "connectivity_options",
                        "tract_count",
                    )
                ):
                    extraction_cfg = dict(optimal_data)
                    # Strip Bayesian result metadata keys that are not valid extraction config fields
                    for k in (
                        "best_parameters",
                        "best_quality_score",
                        "best_qa_score",
                        "all_iterations",
                        "iteration_results",
                        "target_modality",
                        "run_metadata",
                        "manifest",
                    ):
                        extraction_cfg.pop(k, None)
                else:
                    with open(default_cfg_path, "r") as f:
                        extraction_cfg = json.load(f)
            except Exception as e:
                print(f" Error loading configuration files: {e}")
                return 1

            # Merge optimal parameters into the default config
            optimal_params = optimal_data.get("best_parameters", {})
            if not optimal_params:
                print(" 'best_parameters' not found in the optimal config.")
                return 1

            # If the Bayesian results carries a target modality, keep the extraction metric fixed.
            # (This matters when the base config is the global default.)
            if (
                isinstance(optimal_data, dict)
                and optimal_data.get("target_modality")
                and not extraction_cfg.get("connectivity_values")
            ):
                extraction_cfg["connectivity_values"] = [
                    optimal_data["target_modality"]
                ]

            # Update top-level keys like tract_count, and also nested tracking_parameters
            extraction_cfg.update(optimal_params)

            # Ensure tracking_parameters are properly nested if they exist at the top level
            if "tracking_parameters" not in extraction_cfg:
                extraction_cfg["tracking_parameters"] = {}

            for key in [
                "step_size",
                "fa_threshold",
                "min_length",
                "max_length",
                "turning_angle",
            ]:
                if key in extraction_cfg:
                    extraction_cfg["tracking_parameters"][key] = extraction_cfg.pop(key)

            # Move connectivity_threshold into connectivity_options if present
            if "connectivity_threshold" in extraction_cfg:
                extraction_cfg.setdefault("connectivity_options", {})
                extraction_cfg["connectivity_options"]["connectivity_threshold"] = (
                    extraction_cfg.pop("connectivity_threshold")
                )

            out_selected.mkdir(parents=True, exist_ok=True)
            final_config_path = out_selected / "final_extraction_config.json"
            with open(final_config_path, "w") as f:
                json.dump(extraction_cfg, f, indent=2)

            cmd = [
                sys.executable,
                str(root / "scripts" / "run_pipeline.py"),
                "--extraction-config",
                str(final_config_path),
                "--data-dir",
                _abs(args.data_dir),
                "--output",
                str(out_selected),
                "--step",
                "analysis" if args.analysis_only else "all",
            ]
            if args.verbose:
                print(f" Running with merged config: {final_config_path}")
                cmd.append("--verbose")
            if args.quiet:
                cmd.append("--quiet")

        # Validate config before running analysis/apply
        if isinstance(cfg_json, list):
            validate_json_config(str(extraction_cfg_path))
        else:
            validate_json_config(_abs(args.optimal_config))

        print(f" Running: {' '.join(cmd)}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print(" Complete analysis finished successfully!")
            print(f" Results available in: {out_selected}")
            return 0
        except subprocess.CalledProcessError as e:
            print(f" Analysis failed with error code {e.returncode}")
            return e.returncode

    if args.command == "pipeline":
        if args.backend == "mrtrix":
            print(" [MRtrix] 'pipeline' command is currently implemented via 'tune-bayes' or 'tune-grid'.")
            print(" Redirecting to 'tune-bayes'...")
            # Mock args for tune-bayes
            args.command = "tune-bayes"
            args.output_dir = args.output
            args.n_iterations = 30
            args.max_workers = 1
            args.modalities = None
            args.sample_subjects = True
            # Re-run the tune-bayes logic (which is already backend-aware)
            # This is a bit hacky but works for now.
            # Better: refactor the logic into functions.
            # For now, I'll just print instructions.
            print(f" Suggested command: opticonn tune-bayes --backend mrtrix -i {args.data_dir} -o {args.output} --config {args.config}")
            return 0

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

        # Validate config before running pipeline
        if config_path:
            validate_json_config(config_path)

        print(f" Running: {' '.join(cmd)}")
        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print(" Pipeline execution completed!")
            return 0
        except subprocess.CalledProcessError as e:
            print(f" Pipeline failed with error code {e.returncode}")
            return e.returncode

    if args.command == "tune-bayes":
        if args.backend == "mrtrix":
            # Dispatch to mrtrix_tune.py bayes
            cmd = [
                sys.executable,
                str(root / "scripts" / "mrtrix_tune.py"),
                "bayes",
                "--output-dir",
                _abs(args.output_dir),
                "--n-iterations",
                str(args.n_iterations),
            ]
            if args.data_dir:
                if Path(args.data_dir).is_file():
                    cmd += ["--config", _abs(args.data_dir)]
                else:
                    cmd += ["--derivatives-dir", _abs(args.data_dir)]
            if args.subject:
                cmd += ["--subject", args.subject]
            if args.max_workers:
                cmd += ["--nthreads", str(args.max_workers)]
            if args.verbose:
                cmd.append("--verbose")
            if getattr(args, "dry_run", False):
                cmd.append("--dry-run")

            print(f" Running MRtrix Bayesian tuning: {' '.join(cmd)}")
            env = propagate_no_emoji()
            try:
                subprocess.run(cmd, check=True, env=env)
                return 0
            except subprocess.CalledProcessError as e:
                return e.returncode

        # Run Bayesian optimization separately for each requested modality.
        import json as _json

        out_base = Path(_abs(args.output_dir))
        out_base.mkdir(parents=True, exist_ok=True)

        # Load base config to infer default modalities and to write per-modality derived configs
        try:
            base_cfg_path = Path(_abs(args.config))
            base_cfg = _json.loads(base_cfg_path.read_text())
        except Exception as e:
            print(f" Failed to read config: {args.config}")
            print(f"   Error: {e}")
            return 1

        def _split_modalities(items: list[str]) -> list[str]:
            out: list[str] = []
            for tok in items:
                for part in str(tok).split(","):
                    p = part.strip()
                    if p:
                        out.append(p)
            # De-duplicate while preserving order
            seen: set[str] = set()
            uniq: list[str] = []
            for m in out:
                if m not in seen:
                    seen.add(m)
                    uniq.append(m)
            return uniq

        cfg_modalities = base_cfg.get("connectivity_values")
        inferred_default: list[str] = []
        if isinstance(cfg_modalities, list) and cfg_modalities:
            if "qa" in cfg_modalities:
                inferred_default = ["qa"]
            else:
                # Choose first declared modality in config
                first = cfg_modalities[0]
                if isinstance(first, str) and first.strip():
                    inferred_default = [first.strip()]
        if not inferred_default:
            inferred_default = ["qa"]

        modalities = (
            _split_modalities(args.modalities)
            if args.modalities is not None
            else inferred_default
        )
        if not modalities:
            print(" No modalities provided or inferred; nothing to optimize.")
            return 1

        print(f"{_step('TUNE-BAYES')} Starting Bayesian optimization (per modality)...")
        print(f"   Data: {args.data_dir}")
        print(f"   Output base: {out_base}")
        print(_c(f"   Modalities: {', '.join(modalities)}", _BOLD + _MAGENTA))
        print(f"   Iterations per modality: {args.n_iterations}")
        if args.max_workers > 1:
            print(f"   Workers: {args.max_workers} (parallel execution)")

        cfg_dir = out_base / "_configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict = {
            "type": "bayesian_optimization_manifest",
            "base_config": str(base_cfg_path.resolve()),
            "modalities": [],
        }

        env = propagate_no_emoji()
        for modality in modalities:
            # Derive a config that only extracts/scores the requested modality
            derived = dict(base_cfg)
            derived["connectivity_values"] = [modality]

            # Allow environment-based override so containerized runs don't require
            # editing configs that ship with platform-specific defaults.
            dsi_override = os.environ.get("DSI_STUDIO_CMD") or os.environ.get(
                "DSI_STUDIO_PATH"
            )
            if dsi_override:
                derived["dsi_studio_cmd"] = dsi_override

            derived.setdefault("comment", "")
            derived["comment"] = (
                str(derived.get("comment", "")).rstrip()
                + f"\n[opticonn] Derived for tune-bayes modality='{modality}'"
            ).lstrip()
            derived_path = cfg_dir / f"base_config_{modality}.json"
            derived_path.write_text(_json.dumps(derived, indent=2))

            modality_out = out_base / modality
            cmd = [
                sys.executable,
                str(root / "scripts" / "bayesian_optimizer.py"),
                "-i",
                _abs(args.data_dir),
                "-o",
                str(modality_out),
                "--config",
                str(derived_path),
                "--n-iterations",
                str(args.n_iterations),
                "--n-bootstrap",
                str(args.n_bootstrap),
                "--max-workers",
                str(args.max_workers),
            ]
            if args.sample_subjects:
                cmd.append("--sample-subjects")
            if args.verbose:
                cmd.append("--verbose")

            print(_c(f"\n Modality '{modality}':", _BOLD + _GREEN))
            print(f"   Output: {modality_out}")
            print(f"   Config:  {derived_path}")
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(
                    f" Bayesian optimization failed for modality '{modality}' with exit code {e.returncode}"
                )
                return e.returncode

            results_path = modality_out / "bayesian_optimization_results.json"
            manifest["modalities"].append(
                {
                    "modality": modality,
                    "output_dir": str(modality_out.resolve()),
                    "results_file": str(results_path.resolve()),
                    "derived_config": str(derived_path.resolve()),
                }
            )

        manifest_path = out_base / "bayesian_optimization_manifest.json"
        manifest_path.write_text(_json.dumps(manifest, indent=2))
        print(_c("\n Bayesian optimization completed!", _BOLD + _GREEN))
        print(f" Results base: {out_base}")
        print(f" Manifest: {manifest_path}")
        print(_c("\n Next:", _BOLD + _YELLOW))
        print("   opticonn select -i <output_base> --modality <qa|fa|...>")
        return 0

    if args.command == "sensitivity":
        # Run sensitivity analysis
        cmd = [
            sys.executable,
            str(root / "scripts" / "sensitivity_analyzer.py"),
            "-i",
            _abs(args.data_dir),
            "-o",
            _abs(args.output_dir),
            "--config",
            _abs(args.config),
            "--perturbation",
            str(args.perturbation),
        ]
        if args.parameters:
            cmd.extend(["--parameters"] + args.parameters)
        if args.verbose:
            cmd.append("--verbose")

        print(" Starting sensitivity analysis...")
        print(f"   Data: {args.data_dir}")
        print(f"   Output: {args.output_dir}")
        if args.parameters:
            print(f"   Parameters: {', '.join(args.parameters)}")
        else:
            print("   Parameters: All")

        env = propagate_no_emoji()
        try:
            subprocess.run(cmd, check=True, env=env)
            print(" Sensitivity analysis completed!")
            print(f"\n Results available in: {args.output_dir}")
            print("   - sensitivity_analysis_results.json")
            print("   - sensitivity_analysis_plot.png")
            return 0
        except subprocess.CalledProcessError as e:
            print(f" Sensitivity analysis failed with error code {e.returncode}")
            return e.returncode

    print("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
