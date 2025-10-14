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
    from skopt.space import Real, Integer, Categorical
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
            Integer(self.tract_count[0], self.tract_count[1], name='tract_count'),
            Real(self.fa_threshold[0], self.fa_threshold[1], name='fa_threshold'),
            Integer(self.min_length[0], self.min_length[1], name='min_length'),
            Real(self.turning_angle[0], self.turning_angle[1], name='turning_angle'),
            Real(self.step_size[0], self.step_size[1], name='step_size'),
            Real(self.track_voxel_ratio[0], self.track_voxel_ratio[1], name='track_voxel_ratio'),
            Real(self.connectivity_threshold[0], self.connectivity_threshold[1], name='connectivity_threshold', prior='log-uniform'),
        ]

    def get_param_names(self) -> List[str]:
        """Get list of parameter names in order."""
        return [
            'tract_count', 'fa_threshold', 'min_length', 'turning_angle',
            'step_size', 'track_voxel_ratio', 'connectivity_threshold'
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
        verbose: bool = False
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
                "Bayesian optimization requires scikit-optimize.\n"
                "Install with: pip install scikit-optimize"
            )

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.base_config = base_config
        self.param_space = param_space or ParameterSpace()
        self.n_iterations = n_iterations
        self.n_bootstrap_samples = n_bootstrap_samples
        self.verbose = verbose
        # Concurrency control (can be modified before calling optimize)
        self.max_workers = 1
        self._lock = threading.Lock()

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
        config['tract_count'] = int(params['tract_count'])
        
        tracking_params = config.get('tracking_parameters', {})
        tracking_params.update({
            'fa_threshold': float(params['fa_threshold']),
            'min_length': int(params['min_length']),
            'turning_angle': float(params['turning_angle']),
            'step_size': float(params['step_size']),
            'track_voxel_ratio': float(params['track_voxel_ratio']),
        })
        config['tracking_parameters'] = tracking_params

        connectivity_opts = config.get('connectivity_options', {})
        connectivity_opts['connectivity_threshold'] = float(params['connectivity_threshold'])
        config['connectivity_options'] = connectivity_opts

        # Save config
        config_path = self.iterations_dir / f"iteration_{iteration:04d}_config.json"
        with open(config_path, 'w') as f:
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
        logger.info(f"{'='*70}")
        logger.info(f"Testing parameters:")
        for name, value in params.items():
            logger.info(f"  {name:25s} = {value}")

        # Create config for this parameter combination
        config_path = self._create_config_for_params(params, iteration)

        # Create output directory for this iteration
        iter_output = self.iterations_dir / f"iteration_{iteration:04d}"
        iter_output.mkdir(exist_ok=True)

        try:
            # Run pipeline with these parameters using subprocess (no direct import)
            # Simplified bootstrap evaluation
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_pipeline.py"),
                "--data-dir", str(self.data_dir),
                "--output", str(iter_output),
                "--extraction-config", str(config_path),
                "--step", "all",
                "--quiet"
            ]

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
            if 'pure_qa_score' in df.columns:
                mean_qa = float(df['pure_qa_score'].mean())
            elif 'quality_score' in df.columns:
                mean_qa = float(df['quality_score'].mean())
            else:
                logger.warning(f"⚠️  No QA score found for iteration {iteration}")
                return 0.0

            logger.info(f"✅ QA Score: {mean_qa:.4f}")

            # Helper function to convert numpy types to JSON-safe Python types
            def to_json_safe(v):
                """Convert numpy types to native Python types."""
                if hasattr(v, 'dtype'):
                    if v.dtype.kind in 'iu':  # integer types
                        return int(v)
                    elif v.dtype.kind in 'f':  # float types
                        return float(v)
                elif isinstance(v, (list, tuple)):
                    return [to_json_safe(x) for x in v]
                return v

            # Store results (thread-safe for parallel execution)
            result_record = {
                'iteration': iteration,
                'qa_score': float(mean_qa),
                'params': {k: to_json_safe(v) for k, v in params.items()},
                'config_path': str(config_path),
                'output_dir': str(iter_output)
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
            if hasattr(v, 'dtype'):
                if v.dtype.kind in 'iu':  # integer types
                    return int(v)
                elif v.dtype.kind in 'f':  # float types
                    return float(v)
            elif isinstance(v, (list, tuple)):
                return [to_json_safe(x) for x in v]
            return v
        
        progress = {
            'n_iterations': self.n_iterations,
            'completed_iterations': len(self.iteration_results),
            'best_score': float(self.best_score) if self.best_score != -np.inf else None,
            'best_params': {k: to_json_safe(v) for k, v in (self.best_params or {}).items()},
            'all_iterations': self.iteration_results
        }

        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

        logger.info(f"💾 Progress saved to {progress_file}")

    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization.

        Returns:
            Dictionary with optimization results
        """
        logger.info("\n" + "="*70)
        logger.info("🧠 BAYESIAN OPTIMIZATION FOR TRACTOGRAPHY PARAMETERS")
        logger.info("="*70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of iterations: {self.n_iterations}")
        logger.info(f"Parameter space:")
        for name in self.param_space.get_param_names():
            param_range = getattr(self.param_space, name)
            logger.info(f"  {name:25s}: {param_range}")
        logger.info("="*70 + "\n")

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
            result = gp_minimize(
                objective,
                space,
                n_calls=self.n_iterations,
                random_state=42,
                verbose=self.verbose,
                n_random_starts=5  # Start with 5 random samples
            )
            skopt_result_x = result.x
            skopt_result_fun = result.fun
            skopt_result_n_calls = len(result.x_iters)
        else:
            # Parallel execution using Optimizer.ask/tell with ThreadPoolExecutor
            from skopt import Optimizer as SkOptimizer
            
            opt = SkOptimizer(space, random_state=42, n_initial_points=5)
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
                    for _ in range(min(self.max_workers, self.n_iterations - len(self.iteration_results))):
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
        logger.info("\n" + "="*70)
        logger.info("✅ BAYESIAN OPTIMIZATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Best QA Score: {self.best_score:.4f}")
        logger.info(f"Best parameters:")
        for name, value in (self.best_params or {}).items():
            logger.info(f"  {name:25s} = {value}")
        
        # Save final results (ensure all values are JSON serializable)
        def to_json_safe(v):
            """Convert numpy types to native Python types."""
            if hasattr(v, 'dtype'):
                if v.dtype.kind in 'iu':  # integer types
                    return int(v)
                elif v.dtype.kind in 'f':  # float types
                    return float(v)
            elif isinstance(v, (list, tuple)):
                return [to_json_safe(x) for x in v]
            return v
        
        final_results = {
            'optimization_method': 'bayesian',
            'n_iterations': self.n_iterations,
            'max_workers': self.max_workers,
            'best_qa_score': float(self.best_score),
            'best_parameters': {k: to_json_safe(v) for k, v in (self.best_params or {}).items()},
            'skopt_result': {
                'x': [to_json_safe(v) for v in skopt_result_x],
                'fun': float(skopt_result_fun),
                'n_calls': skopt_result_n_calls,
            },
            'all_iterations': self.iteration_results
        }

        results_file = self.output_dir / "bayesian_optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"\n📊 Full results saved to: {results_file}")
        logger.info("="*70 + "\n")

        return final_results


def main():
    """Command line interface for Bayesian optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bayesian optimization for tractography parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Run Bayesian optimization with 30 iterations
  python bayesian_optimizer.py -i data/fib_samples -o results/bayesian_opt \\
      --config configs/base_config.json --n-iterations 30

  # Quick test with 10 iterations
  python bayesian_optimizer.py -i data/fib_samples -o results/quick_test \\
      --config configs/base_config.json --n-iterations 10 --verbose

Bayesian optimization is much more efficient than grid search:
  - Finds optimal parameters in 20-50 evaluations
  - Learns from previous evaluations
  - Focuses on promising parameter regions
  - Handles continuous and discrete parameters
        """
    )

    parser.add_argument(
        '-i', '--data-dir',
        required=True,
        help='Input data directory containing .fz or .fib.gz files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        required=True,
        help='Output directory for optimization results'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Base configuration JSON file'
    )
    parser.add_argument(
        '--n-iterations',
        type=int,
        default=30,
        help='Number of Bayesian optimization iterations (default: 30)'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=3,
        help='Number of bootstrap samples per evaluation (default: 3)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum number of parallel workers for evaluations (default: 1 = sequential). Use 2-4 for parallel execution.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-emoji',
        action='store_true',
        help='Disable emoji in console output'
    )

    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    # Check if scikit-optimize is available
    if not SKOPT_AVAILABLE:
        print("❌ Error: scikit-optimize is not installed")
        print("Install with: pip install scikit-optimize")
        return 1

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Load base configuration
    try:
        with open(args.config, 'r') as f:
            base_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {args.config}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in configuration file: {e}")
        return 1

    # Create optimizer
    optimizer = BayesianOptimizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_config=base_config,
        n_iterations=args.n_iterations,
        n_bootstrap_samples=args.n_bootstrap,
        verbose=args.verbose
    )
    
    # Set max_workers from CLI argument
    optimizer.max_workers = args.max_workers
    
    if optimizer.max_workers > 1:
        logger.info(f"🔄 Parallel execution enabled with {optimizer.max_workers} workers")

    # Run optimization
    try:
        results = optimizer.optimize()
        logger.info("✅ Optimization completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
