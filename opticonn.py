#!/usr/bin/env python3
"""
OptiConn - User-Friendly Brain Connectivity Parameter Optimization
==================================================================

A streamlined two-phase approach to optimize brain connectivity analysis:

PHASE 1: SWEEP - Find Best Parameters
  🔬 Tests parameter combinations on subset of subjects (3-5 subjects)
  🏆 Identifies top 3 candidates based on quality metrics
  ✅ Validates results through bootstrap cross-validation

PHASE 2: APPLY - Full Dataset Analysis
  🎯 Applies optimal parameters to all subjects
  📊 Produces analysis-ready connectivity datasets
  📈 Generates quality reports and visualizations

QUICK START:
    # Install and setup environment
    ./install.sh
    source .venv/bin/activate
    export DSI_STUDIO_CMD=/path/to/dsi_studio

  # Phase 1: Find optimal parameters (quick test with 3 subjects)
  python opticonn.py sweep --config configs/default_sweep.json --data /path/to/subjects

  # Phase 2: Apply to all subjects
  python opticonn.py apply --config results/best_parameters.json --data /path/to/subjects

  # One-shot workflow (both phases)
  python opticonn.py auto --config configs/default_sweep.json --data /path/to/subjects

Author: OptiConn Development Team
Version: 2.0 - User-Friendly Edition
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
from typing import Optional
import uuid


def load_persistent_config() -> None:
    """Load persistent configuration from .opticonn_config file."""
    config_file = Path(__file__).resolve().parent / ".opticonn_config"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DSI_STUDIO_CMD="):
                        # Extract the value (handle quoted values)
                        value = line.split("=", 1)[1].strip('"').strip("'")
                        if value and not os.environ.get("DSI_STUDIO_CMD"):
                            os.environ["DSI_STUDIO_CMD"] = value
                            # Only set from config if the path still exists
                            if not Path(value).exists():
                                print(
                                    f"⚠️  Warning: DSI Studio path from config not found: {value}"
                                )
                                print(
                                    "   Run ./install.sh --dsi-studio /path/to/dsi_studio to update"
                                )
        except Exception:
            # Non-fatal: continue without config
            pass


# Load persistent configuration at startup
load_persistent_config()


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).resolve().parent


def allocate_run_directory(
    base_output: Path, prefix: str, dry_run: bool = False
) -> tuple[Path, str]:
    """Create (or reserve) a unique run directory under the requested output path."""
    base = Path(base_output).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = uuid.uuid4().hex[:6].upper()
    run_id = f"{prefix}_run_{timestamp}_{token}"
    candidate = base / run_id
    suffix = 1
    while candidate.exists():
        candidate = base / f"{run_id}_{suffix:02d}"
        suffix += 1
    if not dry_run:
        candidate.mkdir(parents=True, exist_ok=True)
    else:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate, run_id


def resolve_script(script_name: str) -> Path:
    """Return a script path from the local vendored `scripts/` package.

    OptiConn requires the vendored pipeline pieces to be present under
    `opticonn/scripts/`. This function will raise FileNotFoundError if the
    requested script is not vendored locally. This enforces a strict, local
    only execution contract and prevents accidental runtime reliance on an
    upstream repository.
    """
    root = get_repo_root()
    local = root / "scripts" / script_name
    if local.exists():
        return local
    raise FileNotFoundError(f"Vendored script not found: {local}")


def load_config(config_path: Path) -> dict:
    """Load and validate JSON configuration file, auto-setting DSI_STUDIO_CMD."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Auto-set DSI_STUDIO_CMD from config if not already set
        if "dsi_studio_cmd" in config and not os.environ.get("DSI_STUDIO_CMD"):
            os.environ["DSI_STUDIO_CMD"] = config["dsi_studio_cmd"]
            logger = logging.getLogger("opticonn")
            logger.debug(
                f"Auto-set DSI_STUDIO_CMD from config: {config['dsi_studio_cmd']}"
            )

        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {config_path}")


def setup_logging(
    verbose: bool = False, quiet: bool = False, output_dir: Optional[Path] = None
) -> logging.Logger:
    """Setup comprehensive logging for OptiConn operations."""

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger("opticonn")
    logger.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler()
    if quiet:
        console.setLevel(logging.WARNING)
    elif verbose:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)

    # Clean console format with emojis
    console_fmt = logging.Formatter("%(message)s")
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler for detailed logs
    if output_dir:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = output_dir / f"opticonn_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)

            logger.debug(f"📝 Detailed logs: {log_file}")
        except Exception as e:
            logger.warning(f"⚠️  Could not setup file logging: {e}")

    return logger


def validate_environment() -> tuple[bool, list[str]]:
    """Validate the OptiConn environment setup."""
    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")

    # Check for virtual environment (recommended)
    if not os.environ.get("VIRTUAL_ENV"):
        issues.append(
            "Virtual environment not activated (recommended: source .venv/bin/activate)"
        )

    # Check for DSI Studio
    dsi_cmd = os.environ.get("DSI_STUDIO_CMD")
    if not dsi_cmd:
        issues.append(
            "DSI_STUDIO_CMD not set (export DSI_STUDIO_CMD=/path/to/dsi_studio)"
        )
    elif not Path(dsi_cmd).exists():
        issues.append(f"DSI Studio not found at: {dsi_cmd}")

    # Check for local vendored scripts; warn if missing
    local_scripts = get_repo_root() / "scripts"
    if not local_scripts.exists():
        issues.append(
            "Local vendored 'scripts/' directory not found - vendor required pipeline pieces into opticonn/scripts/"
        )

    return len(issues) == 0, issues


def run_command(
    cmd: list[str], step_name: str, logger: logging.Logger, dry_run: bool = False
) -> bool:
    """Execute a command with proper logging and error handling."""
    logger.info(f"🚀 {step_name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    if dry_run:
        logger.info(f"[DRY-RUN] Would execute: {' '.join(cmd)}")
        return True

    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        # Stream output
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            if line.strip():
                print(line.rstrip())

        return_code = process.wait()

        if return_code == 0:
            logger.info(f"✅ {step_name} completed successfully")
            return True
        else:
            logger.error(f"❌ {step_name} failed (exit code {return_code})")
            return False

    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}")
        return False


def phase1_sweep(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    subjects: int = 3,
    quick: bool = False,
    logger: logging.Logger = None,
    dry_run: bool = False,
) -> bool:
    """
    Phase 1: Parameter Sweep & Quality Assessment

    Tests different parameter combinations on a subset of subjects to find
    the optimal settings for connectivity analysis.
    """
    if logger is None:
        logger = logging.getLogger("opticonn")

    logger.info("🔬 PHASE 1: PARAMETER SWEEP & QUALITY ASSESSMENT")
    logger.info("=" * 60)
    logger.info(f"📁 Data: {data_dir}")
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"👥 Testing with {subjects} subjects")
    logger.info(f"⚙️  Config: {config_path}")

    # Load and validate config (this auto-sets DSI_STUDIO_CMD)
    try:
        config = load_config(config_path)
        logger.debug(f"Loaded configuration with {len(config)} settings")
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        return False
    # If DSI_STUDIO_CMD is still not set, attempt to discover it from Phase 1
    # artifacts: look for selected_parameters.json under the enclosing optimize/*
    if not os.environ.get("DSI_STUDIO_CMD"):
        try:
            cfg_parent = Path(config_path).parent
            # common layout: .../optimize/optimization_results/top3_candidates.json
            # walk up to the 'optimize' directory if present
            optimize_dir = None
            for p in cfg_parent.parents:
                if p.name == "optimize":
                    optimize_dir = p
                    break
            if optimize_dir is None:
                # fallback: try config_path.parent.parent (optimization_results/..)
                cand = Path(config_path).parent.parent
                if cand.exists() and cand.name == "optimize":
                    optimize_dir = cand
            if optimize_dir and optimize_dir.exists():
                for child in optimize_dir.iterdir():
                    sel = child / "selected_parameters.json"
                    if sel.exists():
                        try:
                            with open(sel, "r") as sf:
                                sp = json.load(sf)
                            selcfg = sp.get("selected_config") or sp
                            dsi = selcfg.get("dsi_studio_cmd") or selcfg.get(
                                "dsi_studio"
                            )
                            if dsi:
                                os.environ["DSI_STUDIO_CMD"] = dsi
                                logger.info(f"🔎 Auto-set DSI_STUDIO_CMD from {sel}")
                                break
                        except Exception:
                            continue
        except Exception:
            # best-effort only; continue and let validate_environment report issues
            pass
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False

    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False

    # Count available subjects
    subject_files = list(data_dir.glob("*.fz")) + list(data_dir.glob("*.fib.gz"))
    total_subjects = len(subject_files)

    if total_subjects == 0:
        logger.error(f"❌ No subject files (.fz/.fib.gz) found in {data_dir}")
        return False

    logger.info(f"📊 Found {total_subjects} subjects available")

    if subjects > total_subjects:
        logger.warning(
            f"⚠️  Requested {subjects} subjects, only {total_subjects} available - using all"
        )
        subjects = total_subjects

    # Use requested subject count for the single comprehensive wave
    logger.info(
        f"🏄 Single comprehensive wave: using {subjects} subjects for optimization"
    )

    # Prepare optimization command (uses vendored scripts)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Invoke the vendored optimizer module via -m so `scripts.*` imports work
    optimizer_module = "scripts.cross_validation_bootstrap_optimizer"
    cmd = [
        sys.executable,
        "-m",
        optimizer_module,
        "--data-dir",
        str(data_dir.resolve()),
        "--output-dir",
        str(output_dir),
        "--subjects",
        str(subjects),
        "--extraction-config",
        str(config_path.resolve()),
    ]

    # For single wave mode, we'll let it run the default 2 waves but with more subjects each
    # This gives us cross-validation while using more data per wave

    if quick:
        # Use smaller parameter grid for quick testing
        quick_config = get_repo_root() / "configs" / "quick_sweep.json"
        if quick_config.exists():
            cmd[cmd.index("--extraction-config") + 1] = str(quick_config)
            logger.info("🏃 Quick mode: using reduced parameter grid")

    # Execute parameter sweep
    success = run_command(cmd, "Parameter Optimization", logger, dry_run)

    if success and not dry_run:
        candidates_file = output_dir / "optimize" / "optimization_results" / "top3_candidates.json"
        if not candidates_file.exists():
            logger.error(f"❌ No results found at: {candidates_file}")
            return False
        logger.info("🏆 TOP PARAMETER CANDIDATES (discriminability, higher is better):")
        for i, c in enumerate(json.loads(candidates_file.read_text()), 1):
            logger.info(
                f"  #{i}: {c['atlas']} + {c['connectivity_metric']} | score={c['average_score']:.3f} "
                f"| repeatability={c['repeatability']:.3f} | tract_count={c['parameters']['tract_count']}"
            )
        logger.info(f"📋 Full ranking: {candidates_file.parent / 'ranked_candidates.json'}")
        return True
    return success


def phase2_apply(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    candidate_index: int = 1,
    logger: logging.Logger = None,
    dry_run: bool = False,
) -> bool:
    """
    Phase 2: Apply Optimal Parameters to Full Dataset

    Uses the best parameters from Phase 1 to process all subjects and
    generate analysis-ready connectivity datasets.
    """
    if logger is None:
        logger = logging.getLogger("opticonn")

    logger.info("🎯 PHASE 2: FULL DATASET ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"📁 Data: {data_dir}")
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"🏆 Using candidate #{candidate_index}")
    logger.info(f"📋 Config: {config_path}")

    # Load and validate config (this auto-sets DSI_STUDIO_CMD)
    try:
        config = load_config(config_path)
        logger.debug(f"Loaded configuration with {len(config)} settings")
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        return False
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        return False

    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False

    # Count subjects for full analysis
    subject_files = list(data_dir.glob("*.fz")) + list(data_dir.glob("*.fib.gz"))
    logger.info(f"👥 Processing {len(subject_files)} subjects")

    # Prepare analysis command (uses vendored scripts)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Invoke the vendored hub module via -m so `scripts.*` imports work in subprocess
    hub_module = "scripts.opticonn_hub"
    cmd = [
        sys.executable,
        "-m",
        hub_module,
        "analyze",
        "-i",
        str(data_dir.resolve()),
        "--optimal-config",
        str(config_path.resolve()),
        "-o",
        str(output_dir),
        "--candidate-index",
        str(candidate_index),
    ]

    # Execute full analysis
    success = run_command(cmd, "Full Dataset Analysis", logger, dry_run)

    if success and not dry_run:
        # Check for analysis results
        results_dir = output_dir / "selected" / "03_selection"
        if results_dir.exists():
            analysis_files = list(results_dir.glob("*_analysis_ready.csv"))
            logger.info("📊 ANALYSIS-READY DATASETS:")
            for i, f in enumerate(analysis_files[:5], 1):
                logger.info(f"  {i}. {f.name}")
            if len(analysis_files) > 5:
                logger.info(f"  ... and {len(analysis_files) - 5} more files")

            logger.info(f"📁 Results directory: {results_dir}")
            logger.info("🎉 Ready for statistical analysis!")

        return True

    return success


def auto_workflow(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    subjects: int = 3,
    quick: bool = False,
    logger: logging.Logger = None,
    dry_run: bool = False,
) -> bool:
    """Run complete workflow: Phase 1 → Phase 2 automatically."""

    if logger is None:
        logger = logging.getLogger("opticonn")

    logger.info("🤖 AUTOMATIC WORKFLOW: PHASE 1 → PHASE 2")
    logger.info("=" * 60)

    # Load and validate config (this auto-sets DSI_STUDIO_CMD)
    try:
        config = load_config(config_path)
        logger.debug(f"Loaded configuration with {len(config)} settings")
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        return False

    # Phase 1: Parameter sweep
    phase1_dir = output_dir / "phase1_optimization"
    success1 = phase1_sweep(
        config_path,
        data_dir,
        phase1_dir,
        subjects,
        quick,
        logger=logger,
        dry_run=dry_run,
    )

    if not success1:
        logger.error("❌ Phase 1 failed - stopping workflow")
        return False

    if dry_run:
        logger.info("[DRY-RUN] Phase 1 completed, would proceed to Phase 2")
        return True

    # Find best parameters
    candidates_file = (
        phase1_dir / "optimize" / "optimization_results" / "top3_candidates.json"
    )
    if not candidates_file.exists():
        logger.error("❌ No optimization results found")
        return False

    logger.info("")
    logger.info("✅ Phase 1 completed successfully")
    logger.info("⏳ Starting Phase 2 in 3 seconds...")
    time.sleep(3)

    # Phase 2: Full analysis
    phase2_dir = output_dir / "phase2_analysis"
    success2 = phase2_apply(candidates_file, data_dir, phase2_dir, 1, logger, dry_run)

    if success2:
        logger.info("")
        logger.info("🎉 COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📁 Optimization results: {phase1_dir}")
        logger.info(f"📁 Analysis results: {phase2_dir}")
        logger.info("✨ Your data is ready for statistical analysis!")
        return True
    else:
        logger.error("❌ Phase 2 failed")
        return False


def main() -> int:
    """Main entry point for OptiConn."""

    parser = argparse.ArgumentParser(
        description="OptiConn - User-Friendly Brain Connectivity Parameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

Find optimal parameters (Phase 1):
  python opticonn.py sweep --config configs/default_sweep.json --data /data/subjects

Apply optimal parameters (Phase 2):
  python opticonn.py apply --config results/best_params.json --data /data/subjects  

Complete workflow:
  python opticonn.py auto --config configs/default_sweep.json --data /data/subjects

Quick testing:
  python opticonn.py sweep --data /data/subjects --quick --subjects 2

Environment check:
  python opticonn.py validate

For help with configurations, see configs/ directory.
        """,
    )

    # Global flags
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output with debug information",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (errors/warnings only)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--dsi-studio",
        dest="dsi_studio",
        type=str,
        default=None,
        help="Path to dsi_studio executable (overrides env/config)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Phase 1: Parameter sweep
    sweep_cmd = subparsers.add_parser("sweep", help="Phase 1: Find optimal parameters")
    sweep_cmd.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/default_sweep.json"),
        help="Sweep configuration file",
    )
    sweep_cmd.add_argument(
        "--data",
        "-d",
        required=True,
        type=Path,
        help="Directory with subject .fz/.fib.gz files",
    )
    sweep_cmd.add_argument(
        "--output", "-o", type=Path, default=Path("results"), help="Output directory"
    )
    sweep_cmd.add_argument(
        "--subjects", "-n", type=int, default=3, help="Number of subjects for testing"
    )
    sweep_cmd.add_argument(
        "--quick", action="store_true", help="Quick mode with minimal parameters"
    )
    # Allow users to pass --dry-run after the subcommand
    sweep_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # Phase 2: Apply optimal parameters
    apply_cmd = subparsers.add_parser("apply", help="Phase 2: Apply optimal parameters")
    apply_cmd.add_argument(
        "--config",
        "-c",
        required=True,
        type=Path,
        help="Optimal parameters file (from Phase 1)",
    )
    apply_cmd.add_argument(
        "--data",
        "-d",
        required=True,
        type=Path,
        help="Directory with subject .fz/.fib.gz files",
    )
    apply_cmd.add_argument(
        "--output", "-o", type=Path, default=Path("analysis"), help="Output directory"
    )
    apply_cmd.add_argument(
        "--candidate",
        type=int,
        default=1,
        help="Candidate index (1=best, 2=second best, etc.)",
    )
    apply_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # Auto workflow
    auto_cmd = subparsers.add_parser("auto", help="Complete workflow (Phase 1 → 2)")
    auto_cmd.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/default_sweep.json"),
        help="Sweep configuration file",
    )
    auto_cmd.add_argument(
        "--data",
        "-d",
        required=True,
        type=Path,
        help="Directory with subject .fz/.fib.gz files",
    )
    auto_cmd.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results"),
        help="Base output directory",
    )
    auto_cmd.add_argument(
        "--subjects",
        "-n",
        type=int,
        default=3,
        help="Number of subjects for Phase 1 testing",
    )
    auto_cmd.add_argument("--quick", action="store_true", help="Quick mode for Phase 1")
    auto_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # Environment validation
    subparsers.add_parser("validate", help="Check environment setup")

    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    run_context: Optional[dict] = None
    if args.command in {"sweep", "apply", "auto"}:
        base_output = getattr(args, "output", None)
        if base_output:
            dry_run_flag = getattr(args, "dry_run", False)
            run_dir, run_id = allocate_run_directory(
                base_output, args.command, dry_run_flag
            )
            run_context = {
                "id": run_id,
                "dir": run_dir,
                "base": Path(base_output),
                "dry_run": dry_run_flag,
            }
            setattr(args, "output", run_dir)

    # Setup logging
    log_dir = getattr(args, "output", None)
    if log_dir:
        log_dir = log_dir / "logs"

    logger = setup_logging(args.verbose, args.quiet, log_dir)

    if run_context:
        logger.info(f"🆔 Run ID: {run_context['id']}")
        logger.info(f"📦 Output directory: {run_context['dir']}")
        if not run_context["dry_run"]:
            metadata = {
                "run_id": run_context["id"],
                "created_at": datetime.now().isoformat(),
                "command": args.command,
                "argv": sys.argv[1:],
                "base_output": str(run_context["base"]),
                "run_output": str(run_context["dir"]),
                "dry_run": run_context["dry_run"],
            }
            if hasattr(args, "config") and getattr(args, "config", None) is not None:
                metadata["config"] = str(args.config)
            if hasattr(args, "data") and getattr(args, "data", None) is not None:
                metadata["data"] = str(args.data)
            if (
                hasattr(args, "subjects")
                and getattr(args, "subjects", None) is not None
            ):
                metadata["subjects"] = args.subjects
            if (
                hasattr(args, "candidate")
                and getattr(args, "candidate", None) is not None
            ):
                metadata["candidate_index"] = args.candidate
            try:
                meta_path = run_context["dir"] / "run_metadata.json"
                meta_path.write_text(json.dumps(metadata, indent=2))
            except Exception as meta_err:
                logger.warning(f"⚠️  Could not write run metadata: {meta_err}")

    # Try to auto-load a config (if provided) so that environment variables
    # like DSI_STUDIO_CMD can be set from the JSON before validation.
    try:
        cfg_path_candidate = None
        if hasattr(args, "config") and args.config:
            cfg_path_candidate = Path(args.config)
        # For 'apply' the config may be an optimal-params file; still try to load
        if cfg_path_candidate and cfg_path_candidate.exists():
            try:
                load_config(cfg_path_candidate)
                logger.debug(
                    f"Auto-loaded config for environment from: {cfg_path_candidate}"
                )
            except Exception as e:
                # Non-fatal: continue and let validate_environment report issues
                logger.debug(f"Could not auto-load config {cfg_path_candidate}: {e}")
            # If DSI_STUDIO_CMD is still not set, attempt to find it in Phase1 artifacts
            if not os.environ.get("DSI_STUDIO_CMD"):
                try:
                    cfg_parent = cfg_path_candidate.parent
                    optimize_dir = None
                    for p in cfg_parent.parents:
                        if p.name == "optimize":
                            optimize_dir = p
                            break
                    if optimize_dir is None:
                        cand = cfg_parent.parent
                        if cand.exists() and cand.name == "optimize":
                            optimize_dir = cand
                    if optimize_dir and optimize_dir.exists():
                        for child in optimize_dir.iterdir():
                            sel = child / "selected_parameters.json"
                            if sel.exists():
                                try:
                                    with open(sel, "r") as sf:
                                        sp = json.load(sf)
                                    selcfg = sp.get("selected_config") or sp
                                    dsi = selcfg.get("dsi_studio_cmd") or selcfg.get(
                                        "dsi_studio"
                                    )
                                    if dsi:
                                        os.environ["DSI_STUDIO_CMD"] = dsi
                                        logger.info(
                                            f"🔎 Auto-set DSI_STUDIO_CMD from {sel}"
                                        )
                                        break
                                except Exception:
                                    continue
                except Exception:
                    pass
                # If user passed a --dsi-studio flag, set it and skip discovery
                if args and getattr(args, "dsi_studio", None):
                    os.environ["DSI_STUDIO_CMD"] = args.dsi_studio
                    logger.info(
                        f"🔧 DSI_STUDIO_CMD set from CLI flag: {args.dsi_studio}"
                    )
            else:
                # No explicit config candidate provided; honor CLI flag or try workspace discovery
                if args and getattr(args, "dsi_studio", None):
                    os.environ["DSI_STUDIO_CMD"] = args.dsi_studio
                    logger.info(
                        f"🔧 DSI_STUDIO_CMD set from CLI flag: {args.dsi_studio}"
                    )
                elif not os.environ.get("DSI_STUDIO_CMD"):
                    # Search workspace for any selected_parameters.json under an optimize directory
                    try:
                        repo = get_repo_root()
                        found = False
                        for sel in repo.rglob("optimize/**/selected_parameters.json"):
                            try:
                                with open(sel, "r") as sf:
                                    sp = json.load(sf)
                                selcfg = sp.get("selected_config") or sp
                                dsi = selcfg.get("dsi_studio_cmd") or selcfg.get(
                                    "dsi_studio"
                                )
                                if dsi:
                                    os.environ["DSI_STUDIO_CMD"] = dsi
                                    logger.info(
                                        f"🔎 Auto-set DSI_STUDIO_CMD from {sel}"
                                    )
                                    found = True
                                    break
                            except Exception:
                                continue
                        if not found:
                            logger.debug(
                                "No selected_parameters.json found during workspace discovery"
                            )
                    except Exception:
                        pass
    except Exception:
        # Best-effort only; avoid crashing here
        pass

    # Show banner
    logger.info("🧠 OptiConn v2.0 - Brain Connectivity Parameter Optimization")
    logger.info("=" * 60)

    # Handle validation command
    if args.command == "validate":
        logger.info("🔧 Checking environment setup...")
        is_valid, issues = validate_environment()

        if is_valid:
            logger.info("✅ Environment is properly configured!")
            logger.info("🚀 OptiConn is ready to use")
            return 0
        else:
            logger.error("❌ Environment issues found:")
            for issue in issues:
                logger.error(f"  • {issue}")
            logger.info("")
            logger.info("💡 Setup instructions:")
            logger.info(
                "  1. Run ./install.sh in the opticonn directory to create the local .venv"
            )
            logger.info("  2. source .venv/bin/activate")
            logger.info("  3. export DSI_STUDIO_CMD=/path/to/dsi_studio")
            return 1

    # Validate environment for analysis commands
    is_valid, issues = validate_environment()
    if not is_valid:
        logger.error("❌ Environment not ready:")
        for issue in issues:
            logger.error(f"  • {issue}")
        logger.info("💡 Run 'python opticonn.py validate' for setup help")
        return 1

    # Execute commands
    try:
        if args.command == "sweep":
            success = phase1_sweep(
                args.config,
                args.data,
                args.output,
                args.subjects,
                args.quick,
                logger=logger,
                dry_run=args.dry_run,
            )

            if success and not args.dry_run:
                candidates_file = (
                    args.output
                    / "optimize"
                    / "optimization_results"
                    / "top3_candidates.json"
                )
                logger.info("")
                logger.info("🎯 Next step (Phase 2):")
                logger.info(
                    f"  python opticonn.py apply --config {candidates_file} --data {args.data}"
                )

            return 0 if success else 1

        elif args.command == "apply":
            success = phase2_apply(
                args.config,
                args.data,
                args.output,
                args.candidate,
                logger,
                args.dry_run,
            )
            return 0 if success else 1

        elif args.command == "auto":
            success = auto_workflow(
                args.config,
                args.data,
                args.output,
                args.subjects,
                args.quick,
                logger,
                args.dry_run,
            )
            return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("")
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
    sys.exit(main())
