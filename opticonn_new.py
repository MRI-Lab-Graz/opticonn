#!/usr/bin/env python3
"""
OptiConn - User-Friendly Connectomics Parameter Optimization
============================================================

A two-phase approach to finding optimal brain connectivity analysis parameters:

PHASE 1: Parameter Sweep & QA Validation
- Sweeps through parameter combinations on a subset of subjects
- Identifies top 3 candidate parameter sets based on quality metrics
- Validates stability through bootstrap cross-validation

PHASE 2: Full Dataset Analysis
- Applies the best parameters to all subjects
- Produces analysis-ready connectivity datasets

Usage Examples:
    # Phase 1: Find optimal parameters (quick validation on 3-5 subjects)
    python opticonn_new.py sweep --config configs/sweep_config.json --data /path/to/subjects

    # Phase 2: Apply best parameters to all subjects
    python opticonn_new.py apply --config results/optimization_results/top_candidate.json --data /path/to/subjects

    # One-shot: Run both phases automatically
    python opticonn_new.py auto --config configs/sweep_config.json --data /path/to/subjects

Requirements:
- Virtual environment must be activated (braingraph_pipeline/)
- DSI Studio must be installed and accessible
- Input data should be .fz or .fib.gz files

Author: OptiConn Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
import subprocess
from datetime import datetime


# Ensure we're in the right directory and venv
def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _bootstrap_venv_if_available() -> None:
    """Auto-activate virtual environment if available (same logic as original opticonn.py)"""
    if os.environ.get("OPTICONN_SKIP_VENV", "0") in ("1", "true", "yes"):
        return
    root = _repo_root()

    candidate_venvs = [
        root / "braingraph_pipeline",
        root / ".venv",
        root / "venv",
    ]

    for venv in candidate_venvs:
        # Check for python executable
        posix_py = venv / "bin" / "python"
        win_py = venv / "Scripts" / "python.exe"
        py_path = (
            posix_py if posix_py.exists() else (win_py if win_py.exists() else None)
        )

        if py_path and py_path.exists():
            # Check if we're already in this venv
            ve = os.environ.get("VIRTUAL_ENV")
            if ve and Path(ve).resolve() == venv.resolve():
                return
            try:
                if venv.resolve() in Path(sys.prefix).resolve().parents:
                    return
            except Exception:
                pass

            # Re-exec with venv python
            os.execv(str(py_path), [str(py_path), __file__, *sys.argv[1:]])

    # No venv found - continue with current interpreter


def setup_logging(
    verbose: bool = False, quiet: bool = False, log_dir: Path | None = None
) -> logging.Logger:
    """Setup comprehensive logging for OptiConn operations."""

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create main logger
    logger = logging.getLogger("opticonn")
    logger.setLevel(logging.DEBUG)

    # Console handler with appropriate level
    console_handler = logging.StreamHandler()
    if quiet:
        console_handler.setLevel(logging.WARNING)
    elif verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)

    # Simple console format
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if log directory specified
    if log_dir:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"opticonn_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            logger.info(f"📝 Detailed logs: {log_file}")
        except Exception as e:
            logger.warning(f"⚠️  Could not setup file logging: {e}")

    return logger


def validate_environment() -> bool:
    """Validate that the environment is properly set up."""

    # Check if we're in a virtual environment
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv:
        print("❌ Virtual environment not activated!")
        print("🔧 Please run: source braingraph_pipeline/bin/activate")
        return False

    # Check for DSI Studio command
    dsi_cmd = os.environ.get("DSI_STUDIO_CMD")
    if not dsi_cmd:
        print("⚠️  DSI_STUDIO_CMD environment variable not set")
        print("🔧 Please run: export DSI_STUDIO_CMD=/path/to/dsi_studio")
        print("   (or set it in your config file)")

    # Check for required scripts
    repo_root = _repo_root()
    required_scripts = [
        "scripts/cross_validation_bootstrap_optimizer.py",
        "scripts/run_pipeline.py",
        "scripts/opticonn_hub.py",
    ]

    missing_scripts = []
    for script in required_scripts:
        if not (repo_root / script).exists():
            missing_scripts.append(script)

    if missing_scripts:
        print(f"❌ Missing required scripts: {missing_scripts}")
        return False

    print("✅ Environment validation passed")
    return True


def load_config(config_path: Path) -> dict:
    """Load and validate JSON configuration file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {config_path}")


def count_subjects(data_dir: Path, pattern: str = "*.fz") -> int:
    """Count available subject files in data directory."""
    files = list(data_dir.glob(pattern))
    if not files:
        files = list(data_dir.glob("*.fib.gz"))  # fallback pattern
    return len(files)


def run_subprocess(cmd: list[str], logger: logging.Logger, step_name: str) -> bool:
    """Run a subprocess with real-time output and error handling."""
    logger.info(f"🚀 Starting {step_name}...")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream output in real-time
        output_lines = []
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            if line.strip():
                print(line.rstrip())
                output_lines.append(line)

        return_code = process.wait()

        if return_code == 0:
            logger.info(f"✅ {step_name} completed successfully")
            return True
        else:
            logger.error(f"❌ {step_name} failed with return code {return_code}")
            return False

    except Exception as e:
        logger.error(f"❌ {step_name} failed with exception: {e}")
        return False


def phase1_parameter_sweep(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    subjects: int = 3,
    quick: bool = False,
    logger: logging.Logger = None,
) -> bool:
    """
    Phase 1: Parameter Sweep & Quality Assessment

    Runs parameter optimization on a subset of subjects to identify
    the top 3 candidate parameter combinations.
    """
    if logger is None:
        logger = logging.getLogger("opticonn")

    logger.info("=" * 60)
    logger.info("🔬 PHASE 1: PARAMETER SWEEP & QUALITY ASSESSMENT")
    logger.info("=" * 60)
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"👥 Sample size: {subjects} subjects")
    logger.info(f"⚙️  Configuration: {config_path}")

    # Validate inputs
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False

    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False

    # Count available subjects
    total_subjects = count_subjects(data_dir)
    if total_subjects == 0:
        logger.error(f"❌ No subject files found in {data_dir}")
        return False

    logger.info(f"📊 Found {total_subjects} total subjects")

    if subjects > total_subjects:
        logger.warning(
            f"⚠️  Requested {subjects} subjects, but only {total_subjects} available"
        )
        subjects = total_subjects

    # Prepare command for parameter optimization
    repo_root = _repo_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "cross_validation_bootstrap_optimizer.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--subjects",
        str(subjects),
        "--extraction-config",
        str(config_path),
        "--pareto-report",
    ]

    if quick:
        # Use micro config for quick testing
        micro_config = repo_root / "configs" / "sweep_micro.json"
        if micro_config.exists():
            cmd[cmd.index("--extraction-config") + 1] = str(micro_config)
            logger.info("🏃 Quick mode: using micro sweep configuration")

    # Run parameter optimization
    success = run_subprocess(cmd, logger, "Parameter Optimization")

    if success:
        # Check for results
        results_dir = output_dir / "optimize" / "optimization_results"
        top_candidates = results_dir / "top3_candidates.json"

        if top_candidates.exists():
            logger.info("✅ Phase 1 completed successfully!")
            logger.info(f"📋 Top candidates: {top_candidates}")

            # Display summary of top candidates
            try:
                with open(top_candidates, "r") as f:
                    candidates = json.load(f)

                logger.info("🏆 TOP PARAMETER CANDIDATES:")
                for i, candidate in enumerate(candidates[:3], 1):
                    score = candidate.get(
                        "average_score", candidate.get("score", "N/A")
                    )
                    atlas = candidate.get("atlas", "N/A")
                    metric = candidate.get("connectivity_metric", "N/A")
                    logger.info(f"  #{i}: {atlas} + {metric} (score: {score})")

            except Exception as e:
                logger.warning(f"⚠️  Could not parse top candidates: {e}")

            return True
        else:
            logger.error(f"❌ Results file not found: {top_candidates}")
            return False

    return False


def phase2_full_analysis(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    candidate_index: int = 1,
    logger: logging.Logger = None,
) -> bool:
    """
    Phase 2: Full Dataset Analysis

    Applies the selected optimal parameters to all subjects in the dataset.
    """
    if logger is None:
        logger = logging.getLogger("opticonn")

    logger.info("=" * 60)
    logger.info("🎯 PHASE 2: FULL DATASET ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"⚙️  Using candidate #{candidate_index}")
    logger.info(f"📋 Configuration: {config_path}")

    # Validate inputs
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False

    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False

    # Count all subjects for full analysis
    total_subjects = count_subjects(data_dir)
    logger.info(f"👥 Processing {total_subjects} subjects")

    # Prepare command for full analysis
    repo_root = _repo_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "opticonn_hub.py"),
        "analyze",
        "-i",
        str(data_dir),
        "--optimal-config",
        str(config_path),
        "-o",
        str(output_dir),
        "--candidate-index",
        str(candidate_index),
    ]

    # Run full analysis
    success = run_subprocess(cmd, logger, "Full Dataset Analysis")

    if success:
        logger.info("✅ Phase 2 completed successfully!")

        # Check for analysis-ready results
        results_dir = output_dir / "selected" / "03_selection"
        if results_dir.exists():
            analysis_files = list(results_dir.glob("*_analysis_ready.csv"))
            logger.info(f"📊 Analysis-ready datasets: {len(analysis_files)} files")
            for f in analysis_files[:3]:  # Show first 3
                logger.info(f"  • {f.name}")
            if len(analysis_files) > 3:
                logger.info(f"  • ... and {len(analysis_files) - 3} more")

        return True

    return False


def auto_workflow(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    subjects: int = 3,
    quick: bool = False,
    logger: logging.Logger = None,
) -> bool:
    """
    Automatic workflow: Run both phases sequentially
    """
    if logger is None:
        logger = logging.getLogger("opticonn")

    logger.info("🤖 AUTOMATIC WORKFLOW: PHASE 1 → PHASE 2")
    logger.info("=" * 60)

    # Phase 1: Parameter sweep
    phase1_output = output_dir / "phase1_optimization"
    success1 = phase1_parameter_sweep(
        config_path, data_dir, phase1_output, subjects, quick, logger
    )

    if not success1:
        logger.error("❌ Phase 1 failed - stopping workflow")
        return False

    # Find top candidates file
    top_candidates = (
        phase1_output / "optimize" / "optimization_results" / "top3_candidates.json"
    )
    if not top_candidates.exists():
        logger.error("❌ Top candidates file not found - cannot proceed to Phase 2")
        return False

    # Brief pause and user confirmation (optional)
    logger.info("")
    logger.info("✅ Phase 1 completed. Proceeding to Phase 2...")
    time.sleep(2)

    # Phase 2: Full analysis with best candidate
    phase2_output = output_dir / "phase2_analysis"
    success2 = phase2_full_analysis(
        top_candidates, data_dir, phase2_output, candidate_index=1, logger=logger
    )

    if success2:
        logger.info("=" * 60)
        logger.info("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📁 Phase 1 results: {phase1_output}")
        logger.info(f"📁 Phase 2 results: {phase2_output}")
        return True
    else:
        logger.error("❌ Phase 2 failed")
        return False


def main() -> int:
    """Main entry point for OptiConn."""

    # Auto-activate virtual environment if needed
    _bootstrap_venv_if_available()

    parser = argparse.ArgumentParser(
        description="OptiConn - User-Friendly Connectomics Parameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PHASE-BASED WORKFLOW:

Phase 1 - Parameter Sweep (Find optimal settings):
  python opticonn_new.py sweep --config sweep_config.json --data /data/subjects

Phase 2 - Full Analysis (Apply to all subjects):
  python opticonn_new.py apply --config top_candidate.json --data /data/subjects

Automatic Workflow (Phase 1 → Phase 2):
  python opticonn_new.py auto --config sweep_config.json --data /data/subjects

QUICK TESTING:
  python opticonn_new.py sweep --config sweep_config.json --data /data/subjects --quick
  python opticonn_new.py validate  # Check environment setup

CONFIGURATION FILES:
  • configs/braingraph_default_config.json  - Standard parameters
  • configs/sweep_micro.json               - Quick testing (--quick)
  • Custom sweep configs with parameter ranges for optimization

For detailed documentation, see README.md
        """,
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output (debug level)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (warnings/errors only)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands that would be run without executing",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="OptiConn operation mode"
    )

    # Phase 1: Parameter sweep
    sweep_parser = subparsers.add_parser("sweep", help="Phase 1: Parameter sweep & QA")
    sweep_parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="JSON configuration file with sweep parameters",
    )
    sweep_parser.add_argument(
        "--data",
        "-d",
        required=True,
        type=Path,
        help="Directory containing subject .fz/.fib.gz files",
    )
    sweep_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results)",
    )
    sweep_parser.add_argument(
        "--subjects",
        "-n",
        type=int,
        default=3,
        help="Number of subjects for parameter testing (default: 3)",
    )
    sweep_parser.add_argument(
        "--quick", action="store_true", help="Quick mode: use minimal parameter grid"
    )

    # Phase 2: Full analysis
    apply_parser = subparsers.add_parser(
        "apply", help="Phase 2: Apply optimal parameters"
    )
    apply_parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="JSON file with optimal parameters (from Phase 1)",
    )
    apply_parser.add_argument(
        "--data",
        "-d",
        required=True,
        type=Path,
        help="Directory containing subject .fz/.fib.gz files",
    )
    apply_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("analysis_results"),
        help="Output directory (default: analysis_results)",
    )
    apply_parser.add_argument(
        "--candidate",
        type=int,
        default=1,
        help="Candidate index to use (1=best, 2=second, etc.)",
    )

    # Automatic workflow
    auto_parser = subparsers.add_parser("auto", help="Automatic: Phase 1 → Phase 2")
    auto_parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="JSON configuration file with sweep parameters",
    )
    auto_parser.add_argument(
        "--data",
        "-d",
        required=True,
        type=Path,
        help="Directory containing subject .fz/.fib.gz files",
    )
    auto_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("auto_results"),
        help="Base output directory (default: auto_results)",
    )
    auto_parser.add_argument(
        "--subjects",
        "-n",
        type=int,
        default=3,
        help="Number of subjects for Phase 1 testing (default: 3)",
    )
    auto_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use minimal parameter grid in Phase 1",
    )

    # Environment validation
    _validate_parser = subparsers.add_parser(
        "validate", help="Validate environment setup"
    )

    # Help when no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    # Setup logging
    log_dir = None
    if hasattr(args, "output"):
        log_dir = args.output / "logs"

    logger = setup_logging(args.verbose, args.quiet, log_dir)

    # Display banner
    logger.info("🧠 OptiConn - Brain Connectivity Parameter Optimization")
    logger.info("=" * 60)

    # Handle validation command
    if args.command == "validate":
        logger.info("🔧 Validating environment setup...")
        if validate_environment():
            logger.info("✅ Environment is ready for OptiConn!")
            return 0
        else:
            logger.error("❌ Environment validation failed")
            return 1

    # Validate environment for all other commands
    if not validate_environment():
        logger.error("❌ Environment not properly configured")
        logger.info("💡 Run: python opticonn_new.py validate")
        return 1

    # Handle dry-run mode
    if args.dry_run:
        logger.info("🔍 DRY-RUN MODE: Commands will be shown but not executed")

    # Execute requested command
    try:
        if args.command == "sweep":
            success = phase1_parameter_sweep(
                args.config, args.data, args.output, args.subjects, args.quick, logger
            )

            if success:
                logger.info("💡 Next step: Run Phase 2 with the top candidate")
                top_candidates = (
                    args.output
                    / "optimize"
                    / "optimization_results"
                    / "top3_candidates.json"
                )
                logger.info(
                    f"   python opticonn_new.py apply --config {top_candidates} --data {args.data}"
                )
                return 0
            else:
                return 1

        elif args.command == "apply":
            success = phase2_full_analysis(
                args.config, args.data, args.output, args.candidate, logger
            )
            return 0 if success else 1

        elif args.command == "auto":
            success = auto_workflow(
                args.config, args.data, args.output, args.subjects, args.quick, logger
            )
            return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
