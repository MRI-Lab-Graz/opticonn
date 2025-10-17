#!/usr/bin/env python3
"""
Bayesian Optimization for Tractography Parameters
==================================================

Uses Bayesian optimization to efficiently find optimal tractography parameters
by intelligently sampling the parameter space based on previous evaluations.

This is much more efficient than grid search:
- Grid search: Tests ALL combinations (e.g., 5^6 = 15,625 combinations)
- Bayesian: Finds optimal in 20-50 evaluations

Author: Braingraph Pipeline Team
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import subprocess
import sys
import threading
import concurrent.futures

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt import Optimizer as SkOptimizer

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("⚠️  scikit-optimize not available. Install with: pip install scikit-optimize")

from scripts.utils.runtime import configure_stdio

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Define the parameter space for optimization."""

    tract_count: Tuple[int, int] = (10000, 200000)
    fa_threshold: Tuple[float, float] = (0.05, 0.3)
    min_length: Tuple[int, int] = (5, 50)
    turning_angle: Tuple[float, float] = (30.0, 90.0)
    step_size: Tuple[float, float] = (0.5, 2.0)
    track_voxel_ratio: Tuple[float, float] = (1.0, 5.0)
    connectivity_threshold: Tuple[float, float] = (0.0001, 0.01)

    def to_skopt_space(self) -> List:
        """Convert to scikit-optimize space format."""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")

        return [
            Integer(self.tract_count[0], self.tract_count[1], name="tract_count"),
            Real(self.fa_threshold[0], self.fa_threshold[1], name="fa_threshold"),
            Integer(self.min_length[0], self.min_length[1], name="min_length"),
            Real(self.turning_angle[0], self.turning_angle[1], name="turning_angle"),
            Real(self.step_size[0], self.step_size[1], name="step_size"),
            Real(
                self.track_voxel_ratio[0],
                self.track_voxel_ratio[1],
                name="track_voxel_ratio",
            ),
            Real(
                self.connectivity_threshold[0],
                self.connectivity_threshold[1],
                name="connectivity_threshold",
                prior="log-uniform",
            ),
        ]

    def get_param_names(self) -> List[str]:
        """Get list of parameter names in order."""
        return [
            "tract_count",
            "fa_threshold",
            "min_length",
            "turning_angle",
            "step_size",
            "track_voxel_ratio",
            "connectivity_threshold",
        ]


class BayesianOptimizer:
    """
    Bayesian optimization for tractography parameters.

    Uses Gaussian Process regression to model the relationship between
    parameters and QA scores, then intelligently samples promising regions.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        base_config: Dict,
        param_space: Optional[ParameterSpace] = None,
        n_iterations: int = 30,
        n_bootstrap_samples: int = 3,
        verbose: bool = False,
        max_workers: int = 1,
        random_state: Optional[int] = 42,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            data_dir: Path to input data directory
            output_dir: Path to output directory
            base_config: Base configuration dictionary
            param_space: Parameter space to optimize (uses defaults if None)
            n_iterations: Number of Bayesian optimization iterations
            n_bootstrap_samples: Number of bootstrap samples per evaluation
            verbose: Enable verbose output
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "Bayesian optimization requires scikit-optimize.\n" "Install with: pip install scikit-optimize"
            )

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.base_config = base_config
        self.param_space = param_space or ParameterSpace()
        self.n_iterations = n_iterations
        self.n_bootstrap_samples = n_bootstrap_samples
        self.verbose = verbose
        # Concurrency control (can be modified before calling optimize)
        self.max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        # Random state for reproducibility (pass None for non-deterministic)
        self.random_state = random_state

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iterations_dir = self.output_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)

        # Results storage
        self.iteration_results = []
        self.best_params = None
        self.best_score = -np.inf

    def _create_config_for_params(self, params: Dict[str, Any], iteration: int) -> Path:
        """Create a JSON config file for the given parameters."""
        config = self.base_config.copy()

        # Update with optimized parameters
        config["tract_count"] = int(params["tract_count"])

        tracking_params = config.get("tracking_parameters", {})
        tracking_params.update(
            {
                "fa_threshold": float(params["fa_threshold"]),
                "min_length": int(params["min_length"]),
                "turning_angle": float(params["turning_angle"]),
                "step_size": float(params["step_size"]),
                "track_voxel_ratio": float(params["track_voxel_ratio"]),
            }
        )
        config["tracking_parameters"] = tracking_params

        connectivity_opts = config.get("connectivity_options", {})
        connectivity_opts["connectivity_threshold"] = float(params["connectivity_threshold"])
        config["connectivity_options"] = connectivity_opts

        # Save config
        config_path = self.iterations_dir / f"iteration_{iteration:04d}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return config_path

    def _evaluate_params(self, params_list: List[float], iteration: int) -> float:
        """
        Evaluate a parameter combination by running tractography and computing QA.

        Args:
            params_list: List of parameter values in order
            iteration: Current iteration number

        Returns:
            Negative mean QA score (negative because skopt minimizes)
        """
        # Convert params list to dict
        param_names = self.param_space.get_param_names()
        params = dict(zip(param_names, params_list))

        logger.info(f"\n{'='*70}")
        logger.info(f"🔬 Bayesian Iteration {iteration}/{self.n_iterations}")
        logger.info("=" * 70)
        logger.info("Testing parameters:")
        for name, value in params.items():
            logger.info(f"  {name:25s} = {value}")

        # Create config for this parameter combination
        config_path = self._create_config_for_params(params, iteration)

        # Create output directory for this iteration
        iter_output = self.iterations_dir / f"iteration_{iteration:04d}"
        iter_output.mkdir(exist_ok=True)

        try:
            cmd = [
                sys.executable,
                str(Path(__file__).parent.parent / "subcommands" / "apply.py"),
                "--data-dir",
                str(self.data_dir),
                "--output",
                str(iter_output),
                "--extraction-config",
                str(config_path),
                "--step",
                "all",
            ]
            if self.verbose:
                cmd.append("--verbose")
            else:
                cmd.append("--quiet")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(f"⚠️  Pipeline failed for iteration {iteration}")
                logger.debug(f"stdout: {result.stdout[-500:]}")
                logger.debug(f"stderr: {result.stderr[-500:]}")
                return 0.0  # Return poor score for failed evaluations

            # Extract QA score from results
            opt_csv = iter_output / "02_optimization" / "optimized_metrics.csv"
            if not opt_csv.exists():
                logger.warning(f"⚠️  No optimization results for iteration {iteration}")
                return 0.0

            df = pd.read_csv(opt_csv)

            # Use pure_qa_score if available, otherwise quality_score
            if "pure_qa_score" in df.columns:
                mean_qa = float(df["pure_qa_score"].mean())
            elif "quality_score" in df.columns:
                mean_qa = float(df["quality_score"].mean())
            else:
                logger.warning(f"⚠️  No QA score found for iteration {iteration}")
                return 0.0

            logger.info(f"✅ QA Score: {mean_qa:.4f}")

            # Helper function to convert numpy types to JSON-safe Python types
            def to_json_safe(v):
                """Convert numpy types to native Python types."""
                if hasattr(v, "dtype"):
                    if v.dtype.kind in "iu":  # integer types
                        return int(v)
                    elif v.dtype.kind in "f":  # float types
                        return float(v)
                elif isinstance(v, (list, tuple)):
                    return [to_json_safe(x) for x in v]
                return v

            # Store results (thread-safe for parallel execution)
            result_record = {
                "iteration": iteration,
                "qa_score": float(mean_qa),
                "params": {k: to_json_safe(v) for k, v in params.items()},
                "config_path": str(config_path),
                "output_dir": str(iter_output),
            }

            with self._lock:
                self.iteration_results.append(result_record)

                # Update best
                if mean_qa > self.best_score:
                    self.best_score = mean_qa
                    self.best_params = params.copy()
                    logger.info(f"🏆 New best QA score: {mean_qa:.4f}")

                # Save progress
                self._save_progress()

            # Return negative score (skopt minimizes)
            return -mean_qa

        except Exception as e:
            logger.error(f"❌ Error evaluating iteration {iteration}: {e}")
            return 0.0

    def _save_progress(self):
        """Save optimization progress to JSON."""
        progress_file = self.output_dir / "bayesian_optimization_progress.json"

        # Helper function to convert numpy types to JSON-safe Python types
        def to_json_safe(v):
            """Convert numpy types to native Python types."""
            if hasattr(v, "dtype"):
                if v.dtype.kind in "iu":  # integer types
                    return int(v)
                elif v.dtype.kind in "f":  # float types
                    return float(v)
            elif isinstance(v, (list, tuple)):
                return [to_json_safe(x) for x in v]
            return v

        progress = {
            "n_iterations": self.n_iterations,
            "completed_iterations": len(self.iteration_results),
            "best_score": (float(self.best_score) if self.best_score != -np.inf else None),
            "best_params": {k: to_json_safe(v) for k, v in (self.best_params or {}).items()},
            "all_iterations": self.iteration_results,
        }

        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        logger.info(f"💾 Progress saved to {progress_file}")

    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization.

        Returns:
            Dictionary with optimization results
        """
        logger.info("\n" + "=" * 70)
        logger.info("🧠 BAYESIAN OPTIMIZATION FOR TRACTOGRAPHY PARAMETERS")
        logger.info("=" * 70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of iterations: {self.n_iterations}")
        logger.info("Parameter space:")
        for name in self.param_space.get_param_names():
            param_range = getattr(self.param_space, name)
            logger.info(f"  {name:25s}: {param_range}")
        logger.info("=" * 70 + "\n")

        # Define the objective function for skopt
        space = self.param_space.to_skopt_space()

        @use_named_args(space)
        def objective(**params):
            # Convert params dict to list in correct order
            param_list = [params[name] for name in self.param_space.get_param_names()]
            iteration = len(self.iteration_results) + 1
            return self._evaluate_params(param_list, iteration)

        # Run Bayesian optimization
        logger.info("🚀 Starting Bayesian optimization...\n")

        if self.max_workers > 1:
            logger.info(f"⚡ Running with {self.max_workers} parallel workers\n")

        # Choose sequential or parallel execution
        if self.max_workers <= 1:
            # Sequential execution using gp_minimize
            # Ensure n_random_starts is not greater than n_iterations
            n_random_starts = min(5, self.n_iterations)

            result = gp_minimize(
                objective,
                space,
                n_calls=self.n_iterations,
                random_state=self.random_state,
                verbose=self.verbose,
                n_random_starts=n_random_starts,  # Use calculated value
            )
            skopt_result_x = result.x
            skopt_result_fun = result.fun
            skopt_result_n_calls = len(result.x_iters)
        else:
            # Parallel execution using Optimizer.ask/tell with ThreadPoolExecutor
            # Ensure n_initial_points is not greater than n_iterations
            n_initial_points = min(5, self.n_iterations)
            opt = SkOptimizer(space, random_state=self.random_state, n_initial_points=n_initial_points)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

            # Track next iteration number for parallel execution
            next_iteration = 1
            next_iteration_lock = threading.Lock()

            def evaluate_with_iteration(x):
                """Wrapper to assign iteration numbers in thread-safe manner."""
                nonlocal next_iteration
                with next_iteration_lock:
                    iter_num = next_iteration
                    next_iteration += 1
                return self._evaluate_params(x, iter_num)

            try:
                futures_map = {}  # Map future -> x for tracking

                while len(self.iteration_results) < self.n_iterations:
                    # Ask for new points (up to max_workers at a time)
                    points_to_evaluate = []
                    for _ in range(
                        min(
                            self.max_workers,
                            self.n_iterations - len(self.iteration_results),
                        )
                    ):
                        x = opt.ask()
                        points_to_evaluate.append(x)

                    # Submit evaluations
                    for x in points_to_evaluate:
                        future = executor.submit(evaluate_with_iteration, x)
                        futures_map[future] = x

                    # Collect results and tell optimizer
                    for future in concurrent.futures.as_completed(futures_map.keys()):
                        x = futures_map[future]
                        try:
                            y = future.result()
                            opt.tell(x, y)
                        except Exception as e:
                            logger.error(f"❌ Evaluation failed: {e}")
                            opt.tell(x, 0.0)  # Tell optimizer the evaluation failed
                        finally:
                            del futures_map[future]

            finally:
                executor.shutdown(wait=True)

            # Get best result from our tracked results
            skopt_result_x = list((self.best_params or {}).values()) if self.best_params else []
            skopt_result_fun = -self.best_score if self.best_score != -np.inf else 0.0
            skopt_result_n_calls = len(self.iteration_results)

        # Final results
        logger.info("\n" + "=" * 70)
        logger.info("✅ BAYESIAN OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best QA Score: {self.best_score:.4f}")
        logger.info("Best parameters:")
        for name, value in (self.best_params or {}).items():
            logger.info(f"  {name:25s} = {value}")

        # Save final results (ensure all values are JSON serializable)
        def to_json_safe(v):
            """Convert numpy types to native Python types."""
            if hasattr(v, "dtype"):
                if v.dtype.kind in "iu":  # integer types
                    return int(v)
                elif v.dtype.kind in "f":  # float types
                    return float(v)
            elif isinstance(v, (list, tuple)):
                return [to_json_safe(x) for x in v]
            return v

        final_results = {
            "optimization_method": "bayesian",
            "n_iterations": self.n_iterations,
            "max_workers": self.max_workers,
            "best_qa_score": float(self.best_score),
            "best_parameters": {k: to_json_safe(v) for k, v in (self.best_params or {}).items()},
            "skopt_result": {
                "x": [to_json_safe(v) for v in skopt_result_x],
                "fun": float(skopt_result_fun),
                "n_calls": skopt_result_n_calls,
            },
            "all_iterations": self.iteration_results,
        }

        results_file = self.output_dir / "bayesian_optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"\n📊 Full results saved to: {results_file}")
        logger.info("=" * 70 + "\n")

        return final_results


def main():
    """Command line interface for Bayesian optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Find optimal tractography parameters using Bayesian optimization or sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Find optimal parameters using Bayesian optimization (default)
  opticonn find-optimal-parameters -i data/fib_samples -o results/bayesian_opt \\
      --config configs/base_config.json --n-iterations 30

  # Find optimal parameters using a parameter sweep
  opticonn find-optimal-parameters --method sweep -i data/fib_samples -o results/sweep_opt \\
      --config configs/base_config.json --subjects 5
        """,
    )

    parser.add_argument(
        "--method",
        choices=["bayesian", "sweep"],
        default="bayesian",
        help="Optimization method to use (default: bayesian)",
    )
    parser.add_argument(
        "-i",
        "--data-dir",
        required=True,
        help="Input data directory containing .fz or .fib.gz files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for optimization results",
    )
    parser.add_argument("--config", required=True, help="Base configuration JSON file")
    # Bayesian-specific arguments
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=30,
        help="[Bayesian] Number of optimization iterations (default: 30)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=3,
        help="[Bayesian] Number of bootstrap samples per evaluation (default: 3)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="[Bayesian] Maximum number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="[Bayesian] Random seed for reproducibility (default: 42)",
    )
    # Sweep-specific arguments
    parser.add_argument(
        "--subjects",
        type=int,
        default=3,
        help="[Sweep] Number of subjects for testing (default: 3)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-emoji", action="store_true", help="Disable emoji in console output")

    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    if args.method == "bayesian":
        run_bayesian_optimization(args)
    elif args.method == "sweep":
        run_sweep(args)


def run_bayesian_optimization(args):
    """Runs the Bayesian optimization process."""
    if not SKOPT_AVAILABLE:
        print("❌ Error: scikit-optimize is not installed for Bayesian optimization.")
        print("Install with: pip install scikit-optimize")
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    try:
        with open(args.config, "r") as f:
            base_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in configuration file: {e}")
        sys.exit(1)

    # Extract parameter ranges from config's sweep_parameters
    from scripts.bayesian_optimizer import ParameterSpace
    
    sweep_params = base_config.get('sweep_parameters', {})
    param_space = ParameterSpace(
        tract_count=tuple(sweep_params.get('tract_count_range', [10000, 200000])),
        fa_threshold=tuple(sweep_params.get('fa_threshold_range', [0.05, 0.3])),
        min_length=tuple(sweep_params.get('min_length_range', [5, 50])),
        turning_angle=tuple(sweep_params.get('turning_angle_range', [30.0, 90.0])),
        step_size=tuple(sweep_params.get('step_size_range', [0.5, 2.0])),
        track_voxel_ratio=tuple(sweep_params.get('track_voxel_ratio_range', [1.0, 5.0])),
        connectivity_threshold=tuple(sweep_params.get('connectivity_threshold_range', [0.0001, 0.01]))
    )

    optimizer = BayesianOptimizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_config=base_config,
        param_space=param_space,
        n_iterations=args.n_iterations,
        n_bootstrap_samples=args.n_bootstrap,
        sample_subjects=getattr(args, 'sample_subjects', False),
        verbose=args.verbose,
    )
    
    # Set max_workers if available
    if hasattr(args, 'max_workers'):
        optimizer.max_workers = args.max_workers

    try:
        optimizer.optimize()
        logger.info("✅ Bayesian optimization completed successfully!")
    except Exception as e:
        logger.error(f"❌ Bayesian optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def run_sweep(args):
    """Runs the parameter sweep process."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    logger.info("\n" + "=" * 70)
    logger.info("� PARAMETER SWEEP FOR TRACTOGRAPHY PARAMETERS")
    logger.info("=" * 70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Number of subjects: {args.subjects}")
    logger.info(f"Max parallel workers: {args.max_workers}")
    logger.info("=" * 70 + "\n")

    # Load base configuration
    try:
        with open(args.config, "r") as f:
            base_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in configuration file: {e}")
        sys.exit(1)

    # Import sweep utilities
    try:
        from scripts.utils.sweep_utils import (
            build_param_grid_from_config,
            grid_product,
            random_sampling,
            lhs_sampling,
        )
    except ImportError as e:
        logger.error(f"❌ Failed to import sweep utilities: {e}")
        sys.exit(1)

    # Build parameter grid
    param_values, mapping = build_param_grid_from_config(base_config)

    if not param_values:
        logger.error("❌ No sweep parameters defined in configuration file")
        logger.error("   Add a 'sweep_parameters' section to your config")
        sys.exit(1)

    # Determine sampling strategy
    sweep_params = base_config.get("sweep_parameters", {})
    sampling_config = sweep_params.get("sampling", {})
    method = sampling_config.get("method", "grid").lower()
    n_samples = int(sampling_config.get("n_samples", 0))
    seed = int(sampling_config.get("random_seed", 42))

    # Generate parameter combinations
    if method == "grid" or n_samples <= 0:
        combinations = grid_product(param_values)
        logger.info(f"📊 Using GRID sampling: {len(combinations)} combinations")
    elif method == "random":
        if n_samples <= 0:
            n_samples = 24
        combinations = random_sampling(param_values, n_samples, seed)
        logger.info(f"📊 Using RANDOM sampling: {len(combinations)} combinations")
    elif method == "lhs":
        if n_samples <= 0:
            n_samples = 24
        combinations = lhs_sampling(param_values, n_samples, seed)
        logger.info(f"📊 Using LHS sampling: {len(combinations)} combinations")
    else:
        logger.error(f"❌ Unknown sampling method: {method}")
        sys.exit(1)

    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    combinations_dir = output_dir / "combinations"
    combinations_dir.mkdir(exist_ok=True)

    # Run sweep
    logger.info("\n🚀 Starting parameter sweep...\n")

    sweep_optimizer = SweepOptimizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_config=base_config,
        combinations=combinations,
        mapping=mapping,
        n_subjects=args.subjects,
        max_workers=args.max_workers,
        verbose=args.verbose,
    )

    try:
        results = sweep_optimizer.run()
        logger.info("✅ Parameter sweep completed successfully!")
        return results
    except Exception as e:
        logger.error(f"❌ Parameter sweep failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


class SweepOptimizer:
    """
    Parameter sweep optimizer for tractography parameters.

    Tests multiple parameter combinations and selects the best one.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        base_config: Dict,
        combinations: List[Dict],
        mapping: Dict[str, str],
        n_subjects: int = 3,
        max_workers: int = 1,
        verbose: bool = False,
    ):
        """
        Initialize sweep optimizer.

        Args:
            data_dir: Path to input data directory
            output_dir: Path to output directory
            base_config: Base configuration dictionary
            combinations: List of parameter combinations to test
            mapping: Parameter name to config path mapping
            n_subjects: Number of subjects to use for testing
            max_workers: Maximum number of parallel workers
            verbose: Enable verbose output
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.base_config = base_config
        self.combinations = combinations
        self.mapping = mapping
        self.n_subjects = n_subjects
        self.max_workers = max(1, int(max_workers))
        self.verbose = verbose

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.output_dir / "configs"
        self.configs_dir.mkdir(exist_ok=True)
        self.combinations_dir = self.output_dir / "combinations"
        self.combinations_dir.mkdir(exist_ok=True)

        # Results storage
        self.results = []
        self.best_combination = None
        self.best_score = -np.inf

    def _create_config_for_combination(self, combo: Dict[str, Any], index: int) -> Path:
        """Create a JSON config file for the given parameter combination."""
        # Import sweep utility with fallbacks: absolute -> file-loader
        try:
            from scripts.utils.sweep_utils import apply_param_choice_to_config
        except Exception:
            # Fallback: load module by path (works when package imports are not set up)
            import importlib.util

            sweep_utils_path = Path(__file__).resolve().parents[1] / "utils" / "sweep_utils.py"
            spec = importlib.util.spec_from_file_location("sweep_utils", str(sweep_utils_path))
            sweep_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sweep_utils)
            apply_param_choice_to_config = sweep_utils.apply_param_choice_to_config

        # Apply parameter choices to base config
        config = apply_param_choice_to_config(self.base_config, combo, self.mapping)

        # Add metadata
        config["sweep_meta"] = {
            "index": index,
            "combination": combo,
            "total_combinations": len(self.combinations),
        }

        # Save config
        config_path = self.configs_dir / f"combination_{index:04d}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return config_path

    def _evaluate_combination(self, combo: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Evaluate a parameter combination.

        Args:
            combo: Parameter combination to test
            index: Combination index

        Returns:
            Dictionary with evaluation results
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"🔬 Combination {index}/{len(self.combinations)}")
        logger.info("=" * 70)
        logger.info("Parameters:")
        for key, value in combo.items():
            logger.info(f"  {key:25s} = {value}")

        # Create config for this combination
        config_path = self._create_config_for_combination(combo, index)

        # Create output directory for this combination
        combo_output = self.combinations_dir / f"combination_{index:04d}"
        combo_output.mkdir(exist_ok=True)

        try:
            # Run pipeline with these parameters
            cmd = [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "scripts" / "subcommands" / "apply.py"),
                "--data-dir",
                str(self.data_dir),
                "--output",
                str(combo_output),
                "--extraction-config",
                str(config_path),
                "--step",
                "all",
            ]
            if self.verbose:
                cmd.append("--verbose")
            else:
                cmd.append("--quiet")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(f"⚠️  Pipeline failed for combination {index}")
                if self.verbose:
                    logger.debug(f"stdout: {result.stdout[-500:]}")
                    logger.debug(f"stderr: {result.stderr[-500:]}")
                return {
                    "index": index,
                    "combination": combo,
                    "score": 0.0,
                    "status": "failed",
                }

            # Extract QA score from results
            opt_csv = combo_output / "02_optimization" / "optimized_metrics.csv"
            if not opt_csv.exists():
                logger.warning(f"⚠️  No optimization results for combination {index}")
                return {
                    "index": index,
                    "combination": combo,
                    "score": 0.0,
                    "status": "no_results",
                }

            df = pd.read_csv(opt_csv)

            # Use quality_score_raw mean as primary metric
            if "quality_score_raw" in df.columns:
                score = float(df["quality_score_raw"].mean())
            elif "quality_score" in df.columns:
                score = float(df["quality_score"].mean())
            else:
                logger.warning(f"⚠️  No QA score found for combination {index}")
                return {
                    "index": index,
                    "combination": combo,
                    "score": 0.0,
                    "status": "no_score",
                }

            logger.info(f"✅ QA Score: {score:.4f}")

            return {
                "index": index,
                "combination": combo,
                "score": score,
                "status": "success",
                "config_path": str(config_path),
                "output_dir": str(combo_output),
            }

        except Exception as e:
            logger.error(f"❌ Error evaluating combination {index}: {e}")
            return {
                "index": index,
                "combination": combo,
                "score": 0.0,
                "status": "error",
                "error": str(e),
            }

    def run(self) -> Dict[str, Any]:
        """
        Run parameter sweep.

        Returns:
            Dictionary with sweep results
        """
        import concurrent.futures

        # Evaluate all combinations
        if self.max_workers <= 1:
            # Sequential execution
            for i, combo in enumerate(self.combinations, 1):
                result = self._evaluate_combination(combo, i)
                self.results.append(result)

                if result["score"] > self.best_score:
                    self.best_score = result["score"]
                    self.best_combination = result
        else:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._evaluate_combination, combo, i): i
                    for i, combo in enumerate(self.combinations, 1)
                }

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    self.results.append(result)

                    if result["score"] > self.best_score:
                        self.best_score = result["score"]
                        self.best_combination = result

        # Final results
        logger.info("\n" + "=" * 70)
        logger.info("✅ PARAMETER SWEEP COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best QA Score: {self.best_score:.4f}")
        logger.info(f"Best combination (#{self.best_combination['index']}):")
        for key, value in self.best_combination["combination"].items():
            logger.info(f"  {key:25s} = {value}")

        # Save results
        results_file = self.output_dir / "sweep_results.json"
        final_results = {
            "optimization_method": "sweep",
            "n_combinations": len(self.combinations),
            "best_score": float(self.best_score),
            "best_combination": self.best_combination,
            "all_results": self.results,
        }

        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"\n📊 Full results saved to: {results_file}")
        logger.info("=" * 70 + "\n")

        return final_results


if __name__ == "__main__":
    main()
