#!/usr/bin/env python3
"""
Bayesian Optimization for Tractography Parameters
==================================================

Uses Bayesian optimization to efficiently find optimal tractography parameters
by intelligently sampling the parameter space based on previous evaluations.

This is much more efficient than grid search:
- Grid search: Tests ALL combinations (e.g., 5^6 = 15,625 combinations)
- Bayesian: Finds optimal in 20-50 evaluations

MRI-Lab Graz
Contact: karl.koschutnig@uni-graz.at
GitHub: https://github.com/MRI-Lab-Graz/braingraph-pipeline

Author: Braingraph Pipeline Team
"""

import numpy as np
import pandas as pd
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import subprocess
import sys
import threading
import concurrent.futures
import time

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt import Optimizer as SkOptimizer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("⚠️  scikit-optimize not available. Install with: pip install scikit-optimize")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

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
        sample_subjects: bool = False,
        verbose: bool = False,
        tmp_dir: Optional[str] = None
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            data_dir: Path to input data directory
            output_dir: Path to output directory
            base_config: Base configuration dictionary
            param_space: Parameter space to optimize (uses defaults if None)
            n_iterations: Number of Bayesian optimization iterations
            n_bootstrap_samples: Number of bootstrap samples per evaluation (only used if sample_subjects=False)
            sample_subjects: If True, sample different subject per iteration (recommended for robustness)
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
        self.sample_subjects = sample_subjects
        self.verbose = verbose
        # Concurrency control (can be modified before calling optimize)
        self.max_workers = 1
        self._lock = threading.Lock()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iterations_dir = self.output_dir / "iterations"
        self.iterations_dir.mkdir(exist_ok=True)

        # Temp directory logic
        if tmp_dir is not None:
            self.tmp_dir = Path(tmp_dir)
        else:
            self.tmp_dir = Path("/data/local/tmp_big")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.iteration_results = []
        self.best_params = None
        self.best_score = -np.inf
        self.subjects_used = []  # Track which subjects were actually used
        
        # Get list of available subjects
        self.all_subjects = self._get_all_subjects()
        
        # Select subjects for optimization based on strategy
        if not sample_subjects:
            # Original behavior: select fixed subjects once
            self._select_subjects()
        else:
            # New behavior: will sample different subject per iteration
            self.selected_subjects = []  # Will be populated per iteration
            logger.info(f"📊 Subject sampling mode: different subject per iteration")
            logger.info(f"📊 Total subjects available: {len(self.all_subjects)}")

    def _get_all_subjects(self) -> List[Path]:
        """Get list of all subject files in data directory."""
        all_files = list(self.data_dir.glob("*.fz")) + list(self.data_dir.glob("*.fib.gz"))
        if not all_files:
            logger.warning(f"⚠️  No .fz or .fib.gz files found in {self.data_dir}")
        return all_files

    def _select_subjects(self):
        """Select random subjects for optimization (fixed strategy)."""
        if not self.all_subjects:
            logger.warning(f"⚠️  No subjects available for optimization")
            self.selected_subjects = []
            return
        
        # Select one random subject for the main optimization
        import random
        random.seed(42)  # For reproducibility
        self.selected_subjects = [random.choice(self.all_subjects)]
        
        # Track subject usage
        self.subjects_used = [subj.name for subj in self.selected_subjects]
        
        logger.info(f"📊 Selected primary subject for optimization: {self.selected_subjects[0].name}")
        
        # If bootstrap sampling requested, add additional subjects
        if self.n_bootstrap_samples > 1:
            remaining = [f for f in self.all_subjects if f not in self.selected_subjects]
            if remaining:
                bootstrap_subjects = random.sample(remaining, min(self.n_bootstrap_samples - 1, len(remaining)))
                self.selected_subjects.extend(bootstrap_subjects)
                logger.info(f"📊 Added {len(bootstrap_subjects)} bootstrap subjects")
        
        logger.info(f"📊 Total subjects for optimization: {len(self.selected_subjects)}")
    
    def _sample_subject_for_iteration(self, iteration: int) -> List[Path]:
        """Sample subject(s) for a specific iteration (sampling strategy)."""
        if not self.all_subjects:
            return []
        
        import random
        # Use iteration number as seed for reproducibility
        random.seed(42 + iteration)
        
        # Sample one subject per iteration
        subject = random.choice(self.all_subjects)
        
        # Track subject usage
        if subject.name not in self.subjects_used:
            self.subjects_used.append(subject.name)
        
        logger.info(f"📊 Iteration {iteration}: Sampled subject {subject.name}")
        
        return [subject]

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

        # Validate parameters are within bounds
        for i, (name, value) in enumerate(params.items()):
            param_range = getattr(self.param_space, name)
            if not (param_range[0] <= value <= param_range[1]):
                logger.error(f"❌ Parameter '{name}' = {value} is out of range {param_range}")
                return 0.0  # Return poor score for invalid parameters

        # Determine which subjects to use for this iteration
        if self.sample_subjects:
            # Sample different subject per iteration
            subjects_for_iteration = self._sample_subject_for_iteration(iteration)
        else:
            # Use fixed subjects (original behavior)
            subjects_for_iteration = self.selected_subjects

        logger.info(f"\n{'='*70}")
        logger.info(f"🔬 Bayesian Iteration {iteration}/{self.n_iterations}")
        logger.info(f"{'='*70}")
        logger.info(f"Testing parameters on {len(subjects_for_iteration)} subject(s):")
        for name, value in params.items():
            logger.info(f"  {name:25s} = {value}")

        # Create config for this parameter combination
        config_path = self._create_config_for_params(params, iteration)

        # Create output directory for this iteration
        iter_output = self.iterations_dir / f"iteration_{iteration:04d}"
        iter_output.mkdir(exist_ok=True)
        
        # Create temporary directory with only selected subjects for this iteration
        import tempfile
        import shutil
        temp_data_dir = tempfile.mkdtemp(prefix=f"bayes_iter_{iteration:04d}_", dir=str(self.tmp_dir))
        
        try:
            # Copy only selected subjects to temp directory
            for subject_file in subjects_for_iteration:
                shutil.copy2(subject_file, temp_data_dir)
            
            logger.info(f"📁 Using {len(subjects_for_iteration)} subject(s): {', '.join(f.stem for f in subjects_for_iteration)}")

            # Run pipeline with these parameters using subprocess (no direct import)
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_pipeline.py"),
                "--data-dir", str(temp_data_dir),
                "--output", str(iter_output),
                "--extraction-config", str(config_path),
                "--step", "all"
            ]

            # Show activity spinner during long-running subprocess (suppress in verbose mode to reduce output duplication)
            spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            spinner_idx = 0
            show_spinner = self.verbose  # Only show spinner in verbose mode
            
            def run_with_spinner():
                """Run subprocess and show spinner during execution."""
                nonlocal spinner_idx
                # Set environment variables to redirect all temp/cache files to our temp directory
                env = os.environ.copy()
                env['TMPDIR'] = str(self.tmp_dir)
                env['TEMP'] = str(self.tmp_dir)
                env['TMP'] = str(self.tmp_dir)
                # Enable Qt offscreen mode for DSI Studio on headless servers
                env['QT_QPA_PLATFORM'] = 'offscreen'
                result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                
                # Show spinner while process is running (only if not in quiet mode)
                if show_spinner:
                    while result.poll() is None:
                        sys.stderr.write(f"\r  {spinner_chars[spinner_idx % len(spinner_chars)]} Running... ")
                        sys.stderr.flush()
                        spinner_idx += 1
                        import time
                        time.sleep(0.1)
                    sys.stderr.write("\r  ✓ Complete\n")
                    sys.stderr.flush()
                else:
                    result.wait()  # Wait for completion without spinner
                
                # Get final output
                stdout, stderr = result.communicate()
                
                return result.returncode, stdout, stderr
            
            returncode, stdout, stderr = run_with_spinner()
            
            # Reconstruct result object
            from collections import namedtuple
            Result = namedtuple('Result', ['returncode', 'stdout', 'stderr'])
            result = Result(returncode, stdout, stderr)
            
            if result.returncode != 0:
                logger.warning(f"⚠️  Pipeline failed for iteration {iteration}")
                logger.debug(f"stdout: {result.stdout[-500:]}")
                logger.debug(f"stderr: {result.stderr[-500:]}")
                return 0.0  # Return poor score for failed evaluations

            # Extract QA score from results (with retry for file sync issues in parallel execution)
            opt_csv = iter_output / "02_optimization" / "optimized_metrics.csv"
            if not opt_csv.exists():
                logger.warning(f"⚠️  No optimization results for iteration {iteration}")
                logger.warning(f"   Pipeline stdout: {result.stdout[-1000:]}")
                logger.warning(f"   Pipeline stderr: {result.stderr[-1000:]}")
                return 0.0

            # Retry reading the file to ensure it's fully written (fix for parallel execution race condition)
            import time
            max_retries = 5
            df = None
            for attempt in range(max_retries):
                try:
                    df = pd.read_csv(opt_csv)
                    # Verify the dataframe has expected structure
                    if len(df) > 0:
                        break
                    else:
                        if attempt < max_retries - 1:
                            time.sleep(0.2)
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.debug(f"Retry {attempt+1}/{max_retries} reading {opt_csv.name}: {e}")
                        time.sleep(0.2)
                    else:
                        logger.error(f"Failed to read {opt_csv.name} after {max_retries} attempts: {e}")
                        return 0.0
            
            if df is None or len(df) == 0:
                logger.warning(f"⚠️  No data in optimization results for iteration {iteration}")
                return 0.0
            
            # Use quality_score_raw for unbiased evaluation (normalized quality_score can be artificially inflated)
            if 'quality_score_raw' in df.columns:
                mean_qa = float(df['quality_score_raw'].mean())
            elif 'quality_score' in df.columns:
                logger.warning(f"⚠️  Using normalized quality_score (not ideal - consider using quality_score_raw)")
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
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 0.0
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_data_dir)
            except Exception as e:
                logger.warning(f"⚠️  Could not clean up temp directory {temp_data_dir}: {e}")

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
        import time
        start_time = time.time()
        
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
        
        # Show configuration details
        atlases = self.base_config.get('atlases', [])
        logger.info(f"Atlases: {', '.join(atlases) if atlases else 'None specified'}")
        logger.info("="*70 + "\n")

        # Use Optimizer.ask/tell API for both sequential and parallel execution
        space = self.param_space.to_skopt_space()
        opt = SkOptimizer(space, random_state=42, n_initial_points=5)
        
        if self.max_workers <= 1:
            # Sequential execution with proper progress updates
            try:
                while len(self.iteration_results) < self.n_iterations:
                    x = opt.ask()
                    y = self._evaluate_params(x, len(self.iteration_results) + 1)
                    opt.tell(x, y)
            except KeyboardInterrupt:
                logger.info("⏸️  Optimization interrupted by user")
                raise
            
            skopt_result_x = list((self.best_params or {}).values()) if self.best_params else []
            skopt_result_fun = -self.best_score if self.best_score != -np.inf else 0.0
            skopt_result_n_calls = len(self.iteration_results)
        else:
            # Parallel execution using Optimizer.ask/tell with ThreadPoolExecutor
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
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("\n" + "="*70)
        logger.info("✅ BAYESIAN OPTIMIZATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Best QA Score: {self.best_score:.4f}")
        logger.info(f"Best parameters:")
        for name, value in (self.best_params or {}).items():
            logger.info(f"  {name:25s} = {value}")
        logger.info(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Subjects used: {len(self.subjects_used)} ({', '.join(sorted(self.subjects_used))})")
        logger.info(f"Atlases used: {', '.join(atlases) if atlases else 'None specified'}")
        
        # Next step command
        logger.info("\n" + "🚀 NEXT STEP: Apply optimized parameters")
        logger.info("Run the following command to apply the optimized parameters:")
        logger.info(f"PYTHONPATH=/data/local/software/braingraph-pipeline python scripts/run_pipeline.py \\")
        logger.info(f"  --data-dir /data/local/Poly/derivatives/meta/fz/ \\")
        logger.info(f"  --output optimized_results \\")
        logger.info(f"  --extraction-config {self.output_dir}/iterations/iteration_{self.iteration_results[-1]['iteration']:04d}_config.json \\")
        logger.info(f"  --step all")
        
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
            'total_time_seconds': float(duration),
            'subjects_used': sorted(self.subjects_used),
            'atlases_used': atlases,
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
        default=1,
        help='Number of bootstrap samples per evaluation (default: 1, ignored if --sample-subjects is used)'
    )
    parser.add_argument(
        '--sample-subjects',
        action='store_true',
        help='Sample different subject per iteration (recommended for robust optimization)'
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
    parser.add_argument(
        '--tmp',
        type=str,
        default=None,
        help='Temporary directory for intermediate files (default: /data/local/tmp_big)'
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

    # Extract parameter ranges from config's sweep_parameters
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

    # Create optimizer
    optimizer = BayesianOptimizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_config=base_config,
        param_space=param_space,
        n_iterations=args.n_iterations,
        n_bootstrap_samples=args.n_bootstrap,
        sample_subjects=args.sample_subjects,
        verbose=args.verbose,
        tmp_dir=args.tmp
    )
    
    # Set max_workers from CLI argument
    optimizer.max_workers = args.max_workers
    
    if optimizer.max_workers > 1:
        logger.info(f"🔄 Parallel execution enabled with {optimizer.max_workers} workers")
    
    if args.sample_subjects:
        logger.info(f"🎲 Subject sampling enabled: different subject per iteration")

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
