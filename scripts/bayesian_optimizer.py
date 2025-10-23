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
import time  # noqa: F401

try:
    from skopt import gp_minimize  # noqa: F401
    from skopt.space import Real, Integer, Categorical  # noqa: F401
    from skopt.utils import use_named_args  # noqa: F401
    from skopt import Optimizer as SkOptimizer

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("  scikit-optimize not available. Install with: pip install scikit-optimize")

try:
    from tqdm import tqdm  # noqa: F401

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
        """Convert to scikit-optimize space format, skipping fixed parameters where min==max."""
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")

        space = []

        # Only add parameters with actual ranges (min < max)
        if self.tract_count[0] < self.tract_count[1]:
            space.append(
                Integer(self.tract_count[0], self.tract_count[1], name="tract_count")
            )
        if self.fa_threshold[0] < self.fa_threshold[1]:
            space.append(
                Real(self.fa_threshold[0], self.fa_threshold[1], name="fa_threshold")
            )
        if self.min_length[0] < self.min_length[1]:
            space.append(
                Integer(self.min_length[0], self.min_length[1], name="min_length")
            )
        if self.turning_angle[0] < self.turning_angle[1]:
            space.append(
                Real(self.turning_angle[0], self.turning_angle[1], name="turning_angle")
            )
        if self.step_size[0] < self.step_size[1]:
            space.append(Real(self.step_size[0], self.step_size[1], name="step_size"))
        if self.track_voxel_ratio[0] < self.track_voxel_ratio[1]:
            space.append(
                Real(
                    self.track_voxel_ratio[0],
                    self.track_voxel_ratio[1],
                    name="track_voxel_ratio",
                )
            )
        if self.connectivity_threshold[0] < self.connectivity_threshold[1]:
            space.append(
                Real(
                    self.connectivity_threshold[0],
                    self.connectivity_threshold[1],
                    name="connectivity_threshold",
                    prior="log-uniform",
                )
            )

        if not space:
            raise ValueError(
                " No parameters with ranges found! All parameters are fixed values. Need at least one parameter with a range."
            )

        return space

    def get_param_names(self) -> List[str]:
        """Get list of parameter names that are actually optimized (have ranges)."""
        names = []
        if self.tract_count[0] < self.tract_count[1]:
            names.append("tract_count")
        if self.fa_threshold[0] < self.fa_threshold[1]:
            names.append("fa_threshold")
        if self.min_length[0] < self.min_length[1]:
            names.append("min_length")
        if self.turning_angle[0] < self.turning_angle[1]:
            names.append("turning_angle")
        if self.step_size[0] < self.step_size[1]:
            names.append("step_size")
        if self.track_voxel_ratio[0] < self.track_voxel_ratio[1]:
            names.append("track_voxel_ratio")
        if self.connectivity_threshold[0] < self.connectivity_threshold[1]:
            names.append("connectivity_threshold")
        return names


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
        tmp_dir: Optional[str] = None,
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
            logger.info(" Subject sampling mode: different subject per iteration")
            logger.info(f" Total subjects available: {len(self.all_subjects)}")

    def _get_all_subjects(self) -> List[Path]:
        """Get list of all subject files in data directory."""
        all_files = list(self.data_dir.glob("*.fz")) + list(
            self.data_dir.glob("*.fib.gz")
        )
        if not all_files:
            logger.warning(f"  No .fz or .fib.gz files found in {self.data_dir}")
        return all_files

    def _select_subjects(self):
        """Select random subjects for optimization (fixed strategy)."""
        if not self.all_subjects:
            logger.warning("  No subjects available for optimization")
            self.selected_subjects = []
            return

        # Select one random subject for the main optimization
        import random

        random.seed(42)  # For reproducibility
        self.selected_subjects = [random.choice(self.all_subjects)]

        # Track subject usage
        self.subjects_used = [subj.name for subj in self.selected_subjects]

        logger.info(
            f" Selected primary subject for optimization: {self.selected_subjects[0].name}"
        )

        # If bootstrap sampling requested, add additional subjects
        if self.n_bootstrap_samples > 1:
            remaining = [
                f for f in self.all_subjects if f not in self.selected_subjects
            ]
            if remaining:
                bootstrap_subjects = random.sample(
                    remaining, min(self.n_bootstrap_samples - 1, len(remaining))
                )
                self.selected_subjects.extend(bootstrap_subjects)
                logger.info(f" Added {len(bootstrap_subjects)} bootstrap subjects")

        logger.info(f" Total subjects for optimization: {len(self.selected_subjects)}")

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

        logger.info(f" Iteration {iteration}: Sampled subject {subject.name}")

        return [subject]

    def _create_config_for_params(self, params: Dict[str, Any], iteration: int) -> Path:
        """Create a JSON config file for the given parameters, including fixed ones from param_space."""
        config = self.base_config.copy()

        # Update with optimized parameters
        # Use params dict if available (optimized), otherwise use fixed value from param_space
        tract_count = params.get("tract_count", self.param_space.tract_count[0])
        config["tract_count"] = int(tract_count)

        tracking_params = config.get("tracking_parameters", {})
        tracking_params.update(
            {
                "fa_threshold": float(
                    params.get("fa_threshold", self.param_space.fa_threshold[0])
                ),
                "min_length": int(
                    params.get("min_length", self.param_space.min_length[0])
                ),
                "turning_angle": float(
                    params.get("turning_angle", self.param_space.turning_angle[0])
                ),
                "step_size": float(
                    params.get("step_size", self.param_space.step_size[0])
                ),
                "track_voxel_ratio": float(
                    params.get(
                        "track_voxel_ratio", self.param_space.track_voxel_ratio[0]
                    )
                ),
            }
        )
        config["tracking_parameters"] = tracking_params

        connectivity_opts = config.get("connectivity_options", {})
        connectivity_opts["connectivity_threshold"] = float(
            params.get(
                "connectivity_threshold", self.param_space.connectivity_threshold[0]
            )
        )
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

        # Validate parameters are within bounds
        for i, (name, value) in enumerate(params.items()):
            param_range = getattr(self.param_space, name)
            if not (param_range[0] <= value <= param_range[1]):
                logger.error(
                    f" Parameter '{name}' = {value} is out of range {param_range}"
                )
                return 0.0  # Return poor score for invalid parameters

        # Determine which subjects to use for this iteration
        if self.sample_subjects:
            # Sample different subject per iteration
            subjects_for_iteration = self._sample_subject_for_iteration(iteration)
        else:
            # Use fixed subjects (original behavior)
            subjects_for_iteration = self.selected_subjects

        logger.info(f"\n{'='*70}")
        logger.info(f" Bayesian Iteration {iteration}/{self.n_iterations}")
        logger.info(f"{'='*70}")
        logger.info(f"Testing parameters on {len(subjects_for_iteration)} subject(s):")

        # Log optimized parameters
        for name, value in params.items():
            logger.info(f"  {name:25s} = {value}")

        # Log fixed parameters (those not being optimized)
        fixed_params = []
        for pname in [
            "tract_count",
            "fa_threshold",
            "min_length",
            "turning_angle",
            "step_size",
            "track_voxel_ratio",
            "connectivity_threshold",
        ]:
            if pname not in params:
                prange = getattr(self.param_space, pname)
                if prange[0] == prange[1]:  # Fixed parameter
                    logger.info(f"  {pname:25s} = {prange[0]} (fixed)")
                    fixed_params.append(pname)

        # Create config for this parameter combination
        config_path = self._create_config_for_params(params, iteration)

        # Create output directory for this iteration
        iter_output = self.iterations_dir / f"iteration_{iteration:04d}"
        iter_output.mkdir(exist_ok=True)

        # Create temporary directory with only selected subjects for this iteration
        import tempfile
        import shutil

        temp_data_dir = tempfile.mkdtemp(
            prefix=f"bayes_iter_{iteration:04d}_", dir=str(self.tmp_dir)
        )

        try:
            # Copy only selected subjects to temp directory
            for subject_file in subjects_for_iteration:
                shutil.copy2(subject_file, temp_data_dir)

            logger.info(
                f" Using {len(subjects_for_iteration)} subject(s): {', '.join(f.stem for f in subjects_for_iteration)}"
            )

            # Run pipeline with these parameters using subprocess (no direct import)
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_pipeline.py"),
                "--data-dir",
                str(temp_data_dir),
                "--output",
                str(iter_output),
                "--extraction-config",
                str(config_path),
                "--step",
                "all",
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
                env["TMPDIR"] = str(self.tmp_dir)
                env["TEMP"] = str(self.tmp_dir)
                env["TMP"] = str(self.tmp_dir)
                # Enable Qt offscreen mode for DSI Studio on headless servers
                env["QT_QPA_PLATFORM"] = "offscreen"
                result = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )

                # Show spinner while process is running (only if not in quiet mode)
                if show_spinner:
                    while result.poll() is None:
                        sys.stderr.write(
                            f"\r  {spinner_chars[spinner_idx % len(spinner_chars)]} Running... "
                        )
                        sys.stderr.flush()
                        spinner_idx += 1
                        import time

                        time.sleep(0.1)
                    sys.stderr.write("\r   Complete\n")
                    sys.stderr.flush()
                else:
                    result.wait()  # Wait for completion without spinner

                # Get final output
                stdout, stderr = result.communicate()

                return result.returncode, stdout, stderr

            returncode, stdout, stderr = run_with_spinner()

            # Reconstruct result object
            from collections import namedtuple

            Result = namedtuple("Result", ["returncode", "stdout", "stderr"])
            result = Result(returncode, stdout, stderr)

            if result.returncode != 0:
                logger.warning(f"  Pipeline failed for iteration {iteration}")
                logger.debug(f"stdout: {result.stdout[-500:]}")
                logger.debug(f"stderr: {result.stderr[-500:]}")
                return 0.0  # Return poor score for failed evaluations

            # Extract QA score from results (with retry for file sync issues in parallel execution)
            opt_csv = iter_output / "02_optimization" / "optimized_metrics.csv"
            if not opt_csv.exists():
                logger.warning(f"  No optimization results for iteration {iteration}")
                logger.warning(f"   Pipeline stdout: {result.stdout[-1000:]}")
                logger.warning(f"   Pipeline stderr: {result.stderr[-1000:]}")
                return 0.0
            # Retry reading the file to ensure it's fully written (fix for parallel execution race condition)
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
                        logger.debug(
                            f"Retry {attempt+1}/{max_retries} reading {opt_csv.name}: {e}"
                        )
                        time.sleep(0.2)
                    else:
                        logger.error(
                            f"Failed to read {opt_csv.name} after {max_retries} attempts: {e}"
                        )
                        return 0.0

            if df is None or len(df) == 0:
                logger.warning(
                    f"  No data in optimization results for iteration {iteration}"
                )
                return 0.0

            # ===== CRITICAL: Validate computation integrity =====
            # This prevents faulty results from being used as best scores
            validation_result = self._validate_computation_integrity(
                df, iteration, opt_csv.parent.parent
            )
            if not validation_result["valid"]:
                logger.error(f" COMPUTATION FLAGGED AS FAULTY - Iteration {iteration}")
                logger.error(f"   Reason: {validation_result['reason']}")
                logger.error(f"   Details: {validation_result['details']}")
                # Return neutral score (not the faulty high score)
                # Mark as faulty so it won't contribute to best score

                # Helper function to convert numpy types to JSON-safe Python types
                def to_json_safe(v):
                    """Convert numpy types to native Python types."""
                    if hasattr(v, "dtype"):
                        if hasattr(v, "item"):
                            return v.item()
                        return float(v)
                    if isinstance(v, (list, tuple)):
                        return [to_json_safe(x) for x in v]
                    if isinstance(v, dict):
                        return {k: to_json_safe(val) for k, val in v.items()}
                    return v

                faulty_record = {
                    "iteration": iteration,
                    "qa_score": 0.0,  # Neutral score
                    "params": {k: to_json_safe(v) for k, v in params.items()},
                    "faulty": True,
                    "fault_reason": validation_result["reason"],
                    "config_path": str(config_path),
                    "output_dir": str(iter_output),
                }
                with self._lock:
                    self.iteration_results.append(faulty_record)
                    self._save_progress()
                return 0.0

            # Use quality_score_raw for unbiased evaluation (normalized quality_score can be artificially inflated)
            if "quality_score_raw" in df.columns:
                mean_qa = float(df["quality_score_raw"].mean())
            elif "quality_score" in df.columns:
                logger.warning(
                    "  Using normalized quality_score (not ideal - consider using quality_score_raw)"
                )
                mean_qa = float(df["quality_score"].mean())
            else:
                logger.warning(f"  No QA score found for iteration {iteration}")
                return 0.0

            logger.info(f" QA Score: {mean_qa:.4f} ( Validated)")

            # Store results (thread-safe for parallel execution)
            result_record = {
                "iteration": iteration,
                "qa_score": float(mean_qa),
                "params": {k: to_json_safe(v) for k, v in params.items()},
                "config_path": str(config_path),
                "output_dir": str(iter_output),
                "faulty": False,
            }

            with self._lock:
                self.iteration_results.append(result_record)

                # Update best (only from valid, non-faulty results)
                if mean_qa > self.best_score:
                    self.best_score = mean_qa
                    self.best_params = params.copy()
                    logger.info(f" New best QA score: {mean_qa:.4f}")

                # Save progress
                self._save_progress()

            # Return negative score (skopt minimizes)
            return -mean_qa

        except Exception as e:
            logger.error(f" Error evaluating iteration {iteration}: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            return 0.0

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_data_dir)
            except Exception as e:
                logger.warning(
                    f"  Could not clean up temp directory {temp_data_dir}: {e}"
                )

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
            "best_score": (
                float(self.best_score) if self.best_score != -np.inf else None
            ),
            "best_params": {
                k: to_json_safe(v) for k, v in (self.best_params or {}).items()
            },
            "all_iterations": self.iteration_results,
        }

        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

        logger.info(f" Progress saved to {progress_file}")

    def _validate_computation_integrity(
        self, df: pd.DataFrame, iteration: int, iter_output_dir: Path
    ) -> Dict[str, Any]:
        """
        Validate computation integrity to detect faulty results.

        Checks:
        1. Connectivity matrix generation success (all requested metrics present)
        2. Quality score normalization validity (no artificial 1.0 scores from single subjects)
        3. Expected output files exist and are non-empty
        4. Network metrics are within expected ranges

        Parameters:
        -----------
        df : pd.DataFrame
            Optimization results DataFrame
        iteration : int
            Iteration number
        iter_output_dir : Path
            Iteration output directory

        Returns:
        --------
        Dict with 'valid' (bool), 'reason' (str), 'details' (str)
        """
        # Check 1: Verify extraction logs for connectivity matrix generation success
        extraction_log_file = (
            iter_output_dir / "01_extraction" / "logs" / "extraction_summary.json"
        )
        if extraction_log_file.exists():
            try:
                with open(extraction_log_file, "r") as f:
                    extraction_data = json.load(f)

                # Check if all requested connectivity values were successfully extracted
                if "summary" in extraction_data:
                    summary = extraction_data["summary"]
                    if summary.get("failed", 0) > 0:
                        failed_count = summary["failed"]
                        successful_count = summary["successful"]
                        return {
                            "valid": False,
                            "reason": "Connectivity matrix extraction failure",
                            "details": f"Failed atlases: {failed_count}/{successful_count + failed_count}. Only partial connectivity matrices generated (e.g., count succeeded but FA/QA/NQA failed).",
                        }

                    # Check results detail
                    if "results" in extraction_data:
                        partial_failures = [
                            r
                            for r in extraction_data["results"]
                            if not r.get("success", False)
                        ]
                        if partial_failures:
                            return {
                                "valid": False,
                                "reason": "Partial connectivity extraction",
                                "details": f'Some metrics failed to extract: {[r.get("atlas", "unknown") for r in partial_failures]}',
                            }
            except Exception as e:
                logger.warning(f"Could not verify extraction summary: {e}")

        # Check 2: Quality score sanity checks
        if "quality_score_raw" in df.columns:
            raw_scores = df["quality_score_raw"].values

            # Detect artificial normalization (single value case that was set to 1.0)
            if "quality_score" in df.columns:
                normalized_scores = df["quality_score"].values

                # If raw scores are all identical but normalized is 1.0, this is artificial
                if len(set(raw_scores)) == 1 and all(
                    s >= 0.99 for s in normalized_scores
                ):
                    return {
                        "valid": False,
                        "reason": "Artificial quality score (single subject)",
                        "details": f"All quality scores identical ({raw_scores[0]:.4f}) and normalized to 1.0 - insufficient variation for evaluation",
                    }

        # Check 3: Verify expected files exist
        results_dir = iter_output_dir / "01_extraction" / "results"
        if results_dir.exists():
            atlas_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
            if len(atlas_dirs) == 0:
                return {
                    "valid": False,
                    "reason": "No connectivity results",
                    "details": "Results directory exists but contains no atlas folders - extraction likely failed silently",
                }

            # Check each atlas for actual output files
            for atlas_dir in atlas_dirs:
                connectivity_files = list(atlas_dir.glob("*.connectivity.mat"))
                if len(connectivity_files) == 0:
                    return {
                        "valid": False,
                        "reason": "Missing connectivity matrices",
                        "details": f'Atlas "{atlas_dir.name}" has no .connectivity.mat files - connectivity extraction failed',
                    }

        # Check 4: Network metrics validation
        metric_cols = [
            "density",
            "clustering_coeff_average(weighted)",
            "clustering_coeff_average(binary)",
            "global_efficiency(weighted)",
            "global_efficiency(binary)",
            "small-worldness(weighted)",
            "small-worldness(binary)",
        ]

        for col in metric_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    # Check for NaN or infinite values
                    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                        return {
                            "valid": False,
                            "reason": "Invalid network metrics",
                            "details": f'Column "{col}" contains NaN or infinite values',
                        }

                    # Check for unrealistic values
                    if col == "density":
                        if np.any(values < 0) or np.any(values > 1):
                            return {
                                "valid": False,
                                "reason": "Invalid network metrics",
                                "details": f"Density values outside [0,1] range: {values.min():.4f} to {values.max():.4f}",
                            }

        # All checks passed
        return {"valid": True, "reason": "OK", "details": "All integrity checks passed"}

    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization.

        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info(" BAYESIAN OPTIMIZATION FOR TRACTOGRAPHY PARAMETERS")
        logger.info("=" * 70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of iterations: {self.n_iterations}")
        logger.info("Parameter space:")
        for name in self.param_space.get_param_names():
            param_range = getattr(self.param_space, name)
            logger.info(f"  {name:25s}: {param_range}")

        # Show configuration details
        atlases = self.base_config.get("atlases", [])
        logger.info(f"Atlases: {', '.join(atlases) if atlases else 'None specified'}")
        logger.info("=" * 70 + "\n")

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
                logger.info("  Optimization interrupted by user")
                raise

            skopt_result_x = (
                list((self.best_params or {}).values()) if self.best_params else []
            )
            skopt_result_fun = -self.best_score if self.best_score != -np.inf else 0.0
            skopt_result_n_calls = len(self.iteration_results)
        else:
            # Parallel execution using Optimizer.ask/tell with ThreadPoolExecutor
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            )

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
                            logger.error(f" Evaluation failed: {e}")
                            opt.tell(x, 0.0)  # Tell optimizer the evaluation failed
                        finally:
                            del futures_map[future]

            finally:
                executor.shutdown(wait=True)

            # Get best result from our tracked results
            skopt_result_x = (
                list((self.best_params or {}).values()) if self.best_params else []
            )
            skopt_result_fun = -self.best_score if self.best_score != -np.inf else 0.0
            skopt_result_n_calls = len(self.iteration_results)

        # Final results
        end_time = time.time()
        duration = end_time - start_time

        logger.info("\n" + "=" * 70)
        logger.info(" BAYESIAN OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best QA Score: {self.best_score:.4f}")
        logger.info("Best parameters:")
        for name, value in (self.best_params or {}).items():
            logger.info(f"  {name:25s} = {value}")
        logger.info(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(
            f"Subjects used: {len(self.subjects_used)} ({', '.join(sorted(self.subjects_used))})"
        )
        logger.info(
            f"Atlases used: {', '.join(atlases) if atlases else 'None specified'}"
        )

        # Print comprehensive results summary
        logger.info("\n" + "=" * 85)
        logger.info(" BAYESIAN OPTIMIZATION RESULTS")
        logger.info("=" * 85)

        logger.info("\n Summary:")
        logger.info(f"  Total iterations: {len(self.iteration_results)}")
        logger.info(f"  Best QA Score: {self.best_score:.4f}")
        logger.info(
            f"  Total computation time: {duration:.1f} seconds ({duration/60:.1f} minutes)"
        )

        # Find best iteration details
        best_iter = next(
            (
                r
                for r in self.iteration_results
                if abs(r["qa_score"] - self.best_score) < 0.0001
            ),
            None,
        )

        # Get best atlas for best iteration
        best_atlas = "Unknown"
        best_atlas_qa = 0.0
        if best_iter:
            try:
                iter_dir = (
                    Path(best_iter["output_dir"])
                    / "02_optimization"
                    / "optimized_metrics.csv"
                )
                if iter_dir.exists():
                    import pandas as pd

                    df = pd.read_csv(iter_dir)
                    if "quality_score_raw" in df.columns:
                        best_by_atlas = (
                            df.groupby("atlas")["quality_score_raw"]
                            .max()
                            .sort_values(ascending=False)
                        )
                        if len(best_by_atlas) > 0:
                            best_atlas = best_by_atlas.index[0]
                            best_atlas_qa = best_by_atlas.iloc[0]
            except Exception:
                pass

        if best_iter:
            logger.info(f"\n BEST PARAMETERS (iteration {best_iter['iteration']}):")
            logger.info(f"  Best Atlas: {best_atlas} (QA: {best_atlas_qa:.4f})")
            p = best_iter["params"]
            # Use .get() with fallback to param_space attributes for fixed parameters
            tract_count = p.get("tract_count", self.param_space.tract_count[0])
            fa_threshold = p.get("fa_threshold", self.param_space.fa_threshold[0])
            min_length = p.get("min_length", self.param_space.min_length[0])
            turning_angle = p.get("turning_angle", self.param_space.turning_angle[0])
            step_size = p.get("step_size", self.param_space.step_size[0])
            track_voxel_ratio = p.get(
                "track_voxel_ratio", self.param_space.track_voxel_ratio[0]
            )
            connectivity_threshold = p.get(
                "connectivity_threshold", self.param_space.connectivity_threshold[0]
            )

            logger.info(f"  tract_count            = {int(tract_count):,}")
            logger.info(f"  fa_threshold           = {fa_threshold:.6f}")
            logger.info(f"  min_length             = {int(min_length)}")
            logger.info(f"  turning_angle          = {turning_angle:.2f}°")
            logger.info(f"  step_size              = {step_size:.4f}")
            logger.info(f"  track_voxel_ratio      = {track_voxel_ratio:.4f}")
            logger.info(f"  connectivity_threshold = {connectivity_threshold:.10f}")

        # Show all iterations sorted by QA score
        logger.info("\n ALL ITERATIONS (sorted by QA score):")
        logger.info("-" * 115)
        logger.info(
            f"{'Iter':>4} | {'QA Score':>8} | {'Status':>9} | {'Best Atlas':>30} | {'Atlas QA':>8} | Key Parameters"
        )
        logger.info("-" * 115)

        sorted_iters = sorted(
            self.iteration_results, key=lambda x: x["qa_score"], reverse=True
        )
        for i, result in enumerate(sorted_iters[:20], 1):  # Show top 20
            p = result["params"]
            marker = "" if abs(result["qa_score"] - self.best_score) < 0.0001 else "  "

            # Check if faulty
            is_faulty = result.get("faulty", False)
            status = " FAULTY" if is_faulty else " Valid"

            # Get best atlas and QA for this iteration (if available from extraction results)
            atlas_name = "N/A"
            atlas_qa = "N/A"
            if not is_faulty:
                try:
                    iter_dir = (
                        Path(result["output_dir"])
                        / "02_optimization"
                        / "optimized_metrics.csv"
                    )
                    if iter_dir.exists():
                        import pandas as pd

                        df = pd.read_csv(iter_dir)
                        if "quality_score_raw" in df.columns:
                            best_by_atlas = (
                                df.groupby("atlas")["quality_score_raw"]
                                .max()
                                .sort_values(ascending=False)
                            )
                            if len(best_by_atlas) > 0:
                                atlas_name = best_by_atlas.index[0]
                                atlas_qa = f"{best_by_atlas.iloc[0]:.4f}"
                except Exception:
                    pass
            else:
                atlas_name = result.get("fault_reason", "Extraction failure")

            logger.info(
                f"{marker} {result['iteration']:3d} | {result['qa_score']:8.4f} | {status:>9} | {atlas_name:>30} | {atlas_qa:>8} | tract={int(p.get('tract_count', self.param_space.tract_count[0])):>7,} fa={p.get('fa_threshold', self.param_space.fa_threshold[0]):.3f} angle={p.get('turning_angle', self.param_space.turning_angle[0]):5.1f}°"
            )

        # Show summary of faulty iterations
        faulty_count = sum(1 for r in self.iteration_results if r.get("faulty", False))
        if faulty_count > 0:
            logger.info("-" * 115)
            logger.info(
                f"\n  FAULTY ITERATIONS DETECTED AND FLAGGED: {faulty_count}/{len(self.iteration_results)}"
            )
            logger.info(
                "These iterations were detected as invalid and were NOT used for best score calculation:"
            )
            for result in [r for r in self.iteration_results if r.get("faulty", False)]:
                logger.info(
                    f"  • Iteration {result['iteration']}: {result.get('fault_reason', 'Unknown fault')}"
                )

        logger.info("-" * 115)

        # Next step command
        logger.info("\n" + " NEXT STEP: Apply optimized parameters")
        logger.info(
            "Run the following command to apply the optimized parameters to ALL subjects:"
        )
        logger.info("")
        logger.info(
            "PYTHONPATH=/data/local/software/braingraph-pipeline python scripts/run_pipeline.py \\"
        )
        logger.info("  --data-dir /data/local/Poly/derivatives/meta/fz/ \\")
        logger.info("  --output optimized_results \\")
        logger.info(
            f"  --extraction-config {self.output_dir}/iterations/iteration_{best_iter['iteration'] if best_iter else '0001':04d}_config.json \\"
        )
        logger.info("  --step all")
        logger.info("")

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
            "best_parameters": {
                k: to_json_safe(v) for k, v in (self.best_params or {}).items()
            },
            "total_time_seconds": float(duration),
            "subjects_used": sorted(self.subjects_used),
            "atlases_used": atlases,
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

        logger.info(f"\n Full results saved to: {results_file}")
        logger.info("=" * 70 + "\n")

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
        """,
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
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=30,
        help="Number of Bayesian optimization iterations (default: 30)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1,
        help="Number of bootstrap samples per evaluation (default: 1, ignored if --sample-subjects is used)",
    )
    parser.add_argument(
        "--sample-subjects",
        action="store_true",
        help="Sample different subject per iteration (recommended for robust optimization)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers for evaluations (default: 1 = sequential). Use 2-4 for parallel execution.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--no-emoji", action="store_true", help="Disable emoji in console output"
    )
    parser.add_argument(
        "--tmp",
        type=str,
        default=None,
        help="Temporary directory for intermediate files (default: /data/local/tmp_big)",
    )

    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    # Check if scikit-optimize is available
    if not SKOPT_AVAILABLE:
        print(" Error: scikit-optimize is not installed")
        print("Install with: pip install scikit-optimize")
        return 1

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    # Load and validate configuration using JSONValidator
    from scripts.json_validator import JSONValidator

    validator = JSONValidator()
    is_valid, validation_errors = validator.validate_config(args.config)

    if not is_valid:
        logger.error(f" Configuration validation failed for {args.config}:")
        for error in validation_errors:
            logger.error(f"   • {error}")

        # Print suggestions for fixes
        suggestions = validator.suggest_fixes(args.config)
        if suggestions:
            logger.error("\n Suggested fixes:")
            for suggestion in suggestions:
                logger.error(f"   • {suggestion}")

        return 1

    # Load base configuration (already validated)
    try:
        with open(args.config, "r") as f:
            base_config = json.load(f)
    except FileNotFoundError:
        logger.error(f" Configuration file not found: {args.config}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f" Invalid JSON in configuration file: {e}")
        return 1

    # Extract parameter ranges from config's sweep_parameters
    sweep_params = base_config.get("sweep_parameters", {})

    # Helper function to normalize ranges - handles both [min, max] and [value] formats
    def normalize_range(range_list, is_int=False, default_min=None, default_max=None):
        """Convert range list to (min, max) tuple, handling single values."""
        if not range_list:
            if default_min is not None and default_max is not None:
                return (default_min, default_max)
            raise ValueError("No range provided and no defaults available")

        if len(range_list) == 1:
            # Single value - use it as both min and max (no optimization for this parameter)
            val = range_list[0]
            if is_int:
                val = int(val)
            return (val, val)
        elif len(range_list) >= 2:
            # Range [min, max]
            if is_int:
                return (int(range_list[0]), int(range_list[1]))
            else:
                return (float(range_list[0]), float(range_list[1]))
        else:
            raise ValueError(f"Invalid range format: {range_list}")

    param_space = ParameterSpace(
        tract_count=normalize_range(
            sweep_params.get("tract_count_range", []),
            is_int=True,
            default_min=10000,
            default_max=200000,
        ),
        fa_threshold=normalize_range(
            sweep_params.get("fa_threshold_range", []),
            is_int=False,
            default_min=0.05,
            default_max=0.3,
        ),
        min_length=normalize_range(
            sweep_params.get("min_length_range", []),
            is_int=True,
            default_min=5,
            default_max=50,
        ),
        turning_angle=normalize_range(
            sweep_params.get("turning_angle_range", []),
            is_int=False,
            default_min=30.0,
            default_max=90.0,
        ),
        step_size=normalize_range(
            sweep_params.get("step_size_range", []),
            is_int=False,
            default_min=0.5,
            default_max=2.0,
        ),
        track_voxel_ratio=normalize_range(
            sweep_params.get("track_voxel_ratio_range", []),
            is_int=False,
            default_min=1.0,
            default_max=5.0,
        ),
        connectivity_threshold=normalize_range(
            sweep_params.get("connectivity_threshold_range", []),
            is_int=False,
            default_min=0.0001,
            default_max=0.01,
        ),
    )

    # Validate input data directory BEFORE starting optimization
    data_path = Path(args.data_dir)

    # Check if directory exists
    if not data_path.exists():
        logger.error(f" Data directory does not exist: {args.data_dir}")
        logger.error("   Please create the directory or check the path.")
        return 1

    if not data_path.is_dir():
        logger.error(f" Data path is not a directory: {args.data_dir}")
        return 1

    # Check for .fz or .fib.gz files
    fz_files = list(data_path.glob("*.fz"))
    fib_gz_files = list(data_path.glob("*.fib.gz"))
    all_data_files = fz_files + fib_gz_files

    if not all_data_files:
        logger.error(f" No .fz or .fib.gz files found in: {args.data_dir}")
        logger.error("   Expected to find tractography data files (.fz or .fib.gz)")
        logger.error(f"   Found: {len(list(data_path.glob('*')))} other files")
        if len(list(data_path.glob("*"))) > 0:
            logger.error("   Sample files in directory:")
            for f in sorted(list(data_path.glob("*")))[:5]:
                logger.error(f"     - {f.name}")
        return 1

    logger.info(
        f" Found {len(all_data_files)} data files ({len(fz_files)} .fz, {len(fib_gz_files)} .fib.gz)"
    )

    # Validate iteration count
    if args.n_iterations < 1:
        logger.error(f" Number of iterations must be >= 1, got {args.n_iterations}")
        return 1

    if args.n_iterations > 1000:
        logger.warning(f"  High iteration count: {args.n_iterations} (typical: 20-50)")

    # Validate worker count - STRICT validation
    if args.max_workers < 1:
        logger.error(f" Number of workers must be >= 1, got {args.max_workers}")
        logger.error("   Use --max-workers 1 for sequential execution")
        logger.error("   Use --max-workers 2-8 for parallel execution")
        return 1

    import multiprocessing

    cpu_count = multiprocessing.cpu_count()

    # Don't allow requesting way more workers than CPUs (unless explicitly testing)
    if args.max_workers > cpu_count * 2:
        logger.error(
            f" Requested {args.max_workers} workers but only {cpu_count} CPUs available"
        )
        logger.error(f"   Maximum recommended: {cpu_count} workers (1 per CPU)")
        logger.error(f"   Use --max-workers {cpu_count} for full CPU utilization")
        return 1

    if args.max_workers > cpu_count:
        logger.warning(
            f"  Requested {args.max_workers} workers but only {cpu_count} CPUs available"
        )
        logger.warning(f"   Capping workers to {cpu_count}")
        args.max_workers = cpu_count

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
        tmp_dir=args.tmp,
    )

    # Set max_workers from CLI argument
    optimizer.max_workers = args.max_workers

    if optimizer.max_workers > 1:
        logger.info(f" Parallel execution enabled with {optimizer.max_workers} workers")

    if args.sample_subjects:
        logger.info(" Subject sampling enabled: different subject per iteration")

    # Run optimization
    try:
        optimizer.optimize()
        logger.info(" Optimization completed successfully!")
        return 0
    except Exception as e:
        logger.error(f" Optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
