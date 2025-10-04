#!/usr/bin/env python3
"""
Run Pipeline (Steps 01–03)
==========================

Lightweight orchestrator for the OptiConn pipeline. It runs:

  Step 01: Connectivity extraction (DSI Studio) in batch mode
           → outputs live under <output>/01_connectivity/

  Step 02: Metric optimization on aggregated network measures
           → outputs under <output>/02_optimization/

  Step 03: Optimal selection (prepare datasets for scientific analysis)
           → outputs under <output>/03_selection/

Usage examples:
  # Full run (01→03)
  python run_pipeline.py --step all --data-dir /path/to/fz --output studies/run1 \
         --extraction-config configs/braingraph_default_config.json

  # Analysis only (re-run Step 02→03 using existing Step 01 results in <output>)
  python run_pipeline.py --step analysis --output studies/run1

This script resolves all helper scripts via absolute paths relative to the
repository root, so it is robust to the current working directory and venv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import subprocess


def repo_root() -> Path:
    """Return the repository root directory (where this file lives)."""
    return Path(__file__).resolve().parent


def scripts_dir() -> Path:
    return repo_root() / 'scripts'


def _abs(p: str | os.PathLike | None) -> str | None:
    return None if p is None else str(Path(p).resolve())


def _run(cmd: list[str], cwd: str | None = None, live_prefix: str | None = None) -> int:
    """Run a subprocess with live stdout folding and return code."""
    print(f"🚀 Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    assert proc.stdout is not None
    for line in proc.stdout:
        if live_prefix:
            print(f"[{live_prefix}] {line.rstrip()}")
        else:
            print(line.rstrip())
    return proc.wait()


@dataclass
class Paths:
    output: Path
    step01_dir: Path
    step02_dir: Path
    step03_dir: Path
    agg_csv: Path
    optimized_csv: Path


def build_paths(output_dir: str) -> Paths:
    base = Path(output_dir)
    step01 = base / '01_connectivity'
    step02 = base / '02_optimization'
    step03 = base / '03_selection'
    agg_csv = step01 / 'aggregated_network_measures.csv'
    optimized_csv = step02 / 'optimized_metrics.csv'
    return Paths(base, step01, step02, step03, agg_csv, optimized_csv)


def ensure_dirs(paths: Paths):
    paths.output.mkdir(parents=True, exist_ok=True)
    paths.step01_dir.mkdir(parents=True, exist_ok=True)
    paths.step02_dir.mkdir(parents=True, exist_ok=True)
    paths.step03_dir.mkdir(parents=True, exist_ok=True)


def run_step01(data_dir: str, extraction_config: str, paths: Paths, quiet: bool) -> None:
    """Run batch connectivity extraction (Step 01)."""
    exe = sys.executable
    script = str(scripts_dir() / 'extract_connectivity_matrices.py')
    cmd = [exe, script, '--batch', '-i', data_dir, '-o', str(paths.step01_dir), '--config', extraction_config]
    if quiet:
        cmd.append('--quiet')
    rc = _run(cmd, live_prefix='step01')
    if rc != 0:
        raise SystemExit(f"Step 01 failed with exit code {rc}")


def run_aggregate(paths: Paths) -> None:
    """Aggregate network_measures into a single CSV for optimization."""
    exe = sys.executable
    script = str(scripts_dir() / 'aggregate_network_measures.py')
    cmd = [exe, script, str(paths.step01_dir), str(paths.agg_csv)]
    rc = _run(cmd, live_prefix='aggregate')
    if rc != 0 or not paths.agg_csv.exists():
        raise SystemExit(f"Aggregation failed (code {rc}); expected {paths.agg_csv}")


def run_step02(paths: Paths, quiet: bool) -> None:
    """Run metric optimization (Step 02)."""
    exe = sys.executable
    script = str(scripts_dir() / 'metric_optimizer.py')
    cmd = [exe, script, '-i', str(paths.agg_csv), '-o', str(paths.step02_dir)]
    rc = _run(cmd, live_prefix='step02')
    if rc != 0 or not (paths.step02_dir / 'optimized_metrics.csv').exists():
        raise SystemExit(f"Step 02 failed (code {rc}); expected optimized_metrics.csv in {paths.step02_dir}")


def run_step03(paths: Paths, quiet: bool) -> None:
    """Run optimal selection (Step 03)."""
    exe = sys.executable
    script = str(scripts_dir() / 'optimal_selection.py')
    cmd = [exe, script, '-i', str(paths.optimized_csv), '-o', str(paths.step03_dir)]
    rc = _run(cmd, live_prefix='step03')
    if rc != 0:
        raise SystemExit(f"Step 03 failed with code {rc}")


def maybe_build_extraction_config_from_cv(cv_config_path: str, out_dir: Path) -> str:
    """If a cross-validated dict is provided, derive a minimal extraction config.

    This allows: python run_pipeline.py --cross-validated-config cv.json --step all
    """
    try:
        data = json.loads(Path(cv_config_path).read_text())
        # Expect a dict with atlases/connectivity_values, otherwise no-op
        if isinstance(data, dict):
            atlases = data.get('atlases') or data.get('atlas')
            metrics = data.get('connectivity_values') or data.get('connectivity_metric')
            if isinstance(atlases, str):
                atlases = [atlases]
            if isinstance(metrics, str):
                metrics = [metrics]
            cfg = {
                "description": "Auto-generated from cross-validated config",
                "atlases": atlases or [],
                "connectivity_values": metrics or [],
            }
            out_cfg = out_dir / 'extraction_from_cv.json'
            out_cfg.write_text(json.dumps(cfg, indent=2))
            return str(out_cfg)
    except Exception:
        pass
    return ''


def main() -> int:
    ap = argparse.ArgumentParser(description='OptiConn Pipeline Orchestrator (Steps 01–03)')
    ap.add_argument('--step', default='all', choices=['01', '02', '03', 'all', 'analysis'], help='Which step(s) to run')
    ap.add_argument('-i', '--input', help='Alias for --data-dir (Step 01 input directory) or explicit input for single steps')
    ap.add_argument('--data-dir', help='Input data directory with .fz/.fib.gz files (Step 01)')
    ap.add_argument('--output', required=True, help='Base output directory for pipeline results')
    ap.add_argument('--extraction-config', help='JSON extraction config for Step 01 (default: configs/braingraph_default_config.json)')
    ap.add_argument('--cross-validated-config', help='Optional cross-validated config; will be converted to extraction-config if step includes 01')
    ap.add_argument('--quiet', action='store_true', help='Reduce console output where supported')
    args = ap.parse_args()

    root = repo_root()
    paths = build_paths(args.output)
    ensure_dirs(paths)

    # Normalize data-dir from -i if provided
    if not args.data_dir and args.input:
        args.data_dir = args.input

    # Default extraction config if missing
    extraction_cfg = args.extraction_config
    if not extraction_cfg:
        extraction_cfg = str(root / 'configs' / 'braingraph_default_config.json')

    # If we got a cross-validated config and step includes 01, create a minimal extraction config from it
    if args.cross_validated_config and args.step in ('01', 'all'):
        derived = maybe_build_extraction_config_from_cv(args.cross_validated_config, paths.output)
        if derived:
            extraction_cfg = derived

    t0 = time.time()
    print("==================================================")
    print(f"🧠 OptiConn Pipeline | step={args.step} | output={paths.output}")
    # Echo the extraction configuration being used for transparency
    if args.step in ('01', 'all'):
        try:
            print(f"🔧 Using extraction config: {Path(extraction_cfg).resolve()}")
        except Exception:
            print(f"🔧 Using extraction config: {extraction_cfg}")
    print("==================================================")

    try:
        if args.step in ('01', 'all'):
            if not args.data_dir:
                raise SystemExit("--data-dir (or -i) is required for Step 01")
            run_step01(_abs(args.data_dir), _abs(extraction_cfg), paths, args.quiet)

        if args.step in ('01', 'all', 'analysis', '02', '03'):
            # Ensure aggregated CSV exists for downstream steps
            if args.step in ('all', 'analysis', '02', '03') and not paths.agg_csv.exists():
                run_aggregate(paths)

        if args.step in ('02', 'all', 'analysis'):
            run_step02(paths, args.quiet)

        if args.step in ('03', 'all', 'analysis'):
            # Ensure optimized CSV exists
            if not paths.optimized_csv.exists():
                # If user ran only 03 and provided explicit input file via -i, respect it
                if args.step == '03' and args.input and Path(args.input).exists():
                    paths.optimized_csv = Path(args.input)
                else:
                    raise SystemExit(f"optimized_metrics.csv not found at {paths.optimized_csv}")
            run_step03(paths, args.quiet)

        print("✅ Pipeline completed successfully!")
        print(f"⏱️  Elapsed: {time.time() - t0:.1f}s")
        return 0

    except SystemExit as e:
        # Re-raise explicit exit with message shown above
        if isinstance(e.code, int):
            return e.code
        print(str(e))
        return 1
    except Exception as e:
        print(f"❌ Pipeline crashed: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Braingraph Pipeline Runner
=========================

Simplified pipeline runner that chains the core analysis steps with integrated bootstrap QA validation.
This replaces the complex bash scripts 02-04 with a clean Python workflow.

Usage:
    python run_pipeline.py [--input organized_matrices_dir] [--output results_dir] [--step STEP]

Steps:
    01: Connectivity extraction (extract_connectivity_matrices.py) with JSON config
    02: Network quality optimization (metric_optimizer.py)
    03: Quality-based selection (optimal_selection.py) 
    04: Statistical analysis (deprecated in this package; perform externally)
    all: Run all steps 01-04 (default)
    analysis: Run only steps 02-04 (skip extraction)
    bootstrap-qa: Automatic bootstrap QA validation (integrated)

Examples:
    # Full pipeline with JSON config and bootstrap QA
    python run_pipeline.py --step all --data-dir /path/to/raw/data \\
                          --extraction-config sweep_config.json --enable-bootstrap-qa
    
    # Analysis only (skip extraction)
    python run_pipeline.py --step analysis --input organized_matrices/
    
    # Bootstrap QA validation for production datasets (RECOMMENDED)
    python run_pipeline.py --test-config test_all_subjects.json --enable-bootstrap-qa

Author: Braingraph Pipeline Team
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
import subprocess
from datetime import datetime

def setup_logging(verbose: bool = False, quiet: bool = False, log_dir: str | None = None):
    """Set up logging configuration.

    Writes pipeline_run_YYYYMMDD_HHMMSS.log into the provided log_dir (typically the pipeline
    output directory). Falls back to current working directory if log_dir is not provided.

    Console is concise by default (INFO) or very quiet (WARNING) when --quiet.
    File logs always capture DEBUG for full reproducibility.
    """
    # Decide log file destination
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(log_dir) / f"pipeline_run_{timestamp}.log"
        else:
            log_path = Path(f"pipeline_run_{timestamp}.log")
    except Exception:
        # As a safety net, if creating the directory fails, fall back to CWD
        log_path = Path(f"pipeline_run_{timestamp}.log")

    # Root logger at DEBUG; configure per-handler levels
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicates when called multiple times
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # File handler always at DEBUG for full details
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)

def run_step(script_name, args, logger, step_name, quiet=False):
    """Run a pipeline step and handle errors."""
    logger.info(f"🚀 Starting {step_name}...")
    # Log the exact command at DEBUG for reproducibility (captured in file logs)
    logger.debug(f"Command: python {script_name} {' '.join(args)}")
    # Support dry-run by short-circuiting execution
    dry_run = getattr(args, 'dry_run', False)
    if dry_run:
        logger.info(f"[DRY-RUN] Would run: python {script_name} {' '.join(args)}")
        return True

    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            ['python', script_name] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,  # Unbuffered
            universal_newlines=True
        )

        # Print output in real-time
        output_lines = []
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ''):
            if line.strip():  # Only print non-empty lines
                if not quiet:
                    print(line.rstrip(), flush=True)
                output_lines.append(line)

        # Wait for completion
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

def find_organized_matrices(base_dir):
    """Find the organized matrices directory from step 01."""
    base_path = Path(base_dir)
    
    # Look for organized_matrices directory
    organized_dirs = list(base_path.glob("**/organized_matrices"))
    if organized_dirs:
        return organized_dirs[0]
    
    # Look for directories with atlas structure
    for dir_path in base_path.iterdir():
        if dir_path.is_dir():
            # Check if it has atlas subdirectories
            atlas_dirs = [d for d in dir_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if len(atlas_dirs) > 3:  # Likely contains multiple atlases
                return dir_path
    
    return None

def load_test_configuration(test_config_path):
    """Load and process JSON test configuration."""
    
    # Optional: Validate configuration first
    try:
        from scripts.json_validator import validate_config_file
        if not validate_config_file(test_config_path):
            raise ValueError(f"Configuration validation failed: {test_config_path}")
    except ImportError:
        logging.warning("scripts/json_validator.py not available - skipping validation")
    
    with open(test_config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"🧪 Loading test configuration: {test_config_path}")
    
    # Test config basic info
    test_info = config.get('test_config', {})
    if not test_info.get('enabled', True):
        raise ValueError("Test configuration is disabled")
    
    logging.info(f"📋 Test: {test_info.get('name', 'Unnamed Test')}")
    logging.info(f"📝 Description: {test_info.get('description', 'No description')}")
    
    # File selection logic
    data_selection = config.get('data_selection', {})
    source_dir = data_selection.get('source_dir', 'data')
    method = data_selection.get('selection_method', 'random')
    count = data_selection.get('n_subjects', 3)
    seed = data_selection.get('random_seed', 42)
    file_pattern = data_selection.get('file_pattern', '*.fz')  # Updated for DSI Studio .fz files
    
    # Set random seed for reproducible results
    if seed:
        random.seed(seed)
        logging.info(f"🎲 Random seed set to: {seed}")
    
    # Find available data files
    data_path = Path(source_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {source_dir}")
    
    # Get all files matching pattern
    all_files = list(data_path.glob(file_pattern))

def load_cross_validated_configuration(config_path):
    """Load and process cross-validated optimization configuration."""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"🔬 Loading cross-validated configuration: {config_path}")
    
    # Check if this is a cross-validated config
    if not config.get('cross_validation_optimized', False):
        raise ValueError("Configuration is not a cross-validated optimization result")
    
    # Extract key information
    optimal_params = config.get('optimal_parameters', {})
    data_dir = config.get('data_directory')
    output_dir = config.get('output_directory', 'analysis_results')
    validation_results = config.get('validation_results', {})
    
    logging.info(f"✅ Cross-validation status: {'PASSED' if validation_results.get('validation_passed') else 'FAILED'}")
    logging.info(f"📊 Parameter consistency: {validation_results.get('consistency_score', 0):.1%}")
    logging.info(f"📁 Data directory: {data_dir}")
    logging.info(f"📁 Output directory: {output_dir}")
    logging.info(f"⚙️  Optimal parameters: {optimal_params}")
    
    if not validation_results.get('validation_passed', False):
        raise ValueError("Cross-validation failed - cannot proceed with unvalidated parameters")
    
    if not data_dir or not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Create a synthetic test configuration that the pipeline can understand
    synthetic_config = {
        "test_config": {
            "name": "cross_validated_analysis",
            "description": f"Full analysis with cross-validated parameters (consistency: {validation_results.get('consistency_score', 0):.1%})",
            "enabled": True
        },
        "data_selection": {
            "source_dir": data_dir,
            "selection_method": "all",  # Use all subjects for full analysis
            "file_pattern": "*.fz"
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03", "04"],
            "output_base_dir": output_dir,
            "extraction_config": "cross_validated_extraction_config.json"  # Will be created
        },
        "cross_validation_metadata": {
            "optimal_parameters": optimal_params,
            "validation_results": validation_results,
            "original_config_file": config_path
        }
    }
    
    # Create extraction config with optimal parameters
    extraction_config = {
        "description": "Cross-validated optimal parameters for connectivity extraction",
        "dsi_studio_cmd": "/Applications/dsi_studio.app/Contents/MacOS/dsi_studio",
        "atlases": ["FreeSurferDKT_Cortical", "FreeSurferDKT_Subcortical", "HCP-MMP"],
        "connectivity_values": ["count", "fa", "qa", "ncount2"],
        "tract_count": optimal_params.get("track_count", 1000000),
        "step_size": optimal_params.get("step_size", 0.5),
        "turning_angle": optimal_params.get("angular_threshold", 45),
        "fa_threshold": optimal_params.get("fa_threshold", 0.15),
        "cross_validated": True,
        "validation_metadata": validation_results
    }
    
    # Save extraction config
    extraction_config_file = "cross_validated_extraction_config.json"
    with open(extraction_config_file, 'w') as f:
        json.dump(extraction_config, f, indent=2)
    
    logging.info(f"📝 Created extraction config: {extraction_config_file}")
    
    # Process the synthetic config like a regular test config
    return load_test_configuration_from_dict(synthetic_config)

def load_test_configuration_from_dict(config):
    """Load test configuration from a dictionary (used by cross-validated configs)."""
    
    # Test config basic info
    test_info = config.get('test_config', {})
    if not test_info.get('enabled', True):
        raise ValueError("Test configuration is disabled")
    
    logging.info(f"📋 Test: {test_info.get('name', 'Unnamed Test')}")
    logging.info(f"📝 Description: {test_info.get('description', 'No description')}")
    
    # File selection logic
    data_selection = config.get('data_selection', {})
    source_dir = data_selection.get('source_dir', 'data')
    method = data_selection.get('selection_method', 'random')
    count = data_selection.get('n_subjects', 3)
    seed = data_selection.get('random_seed', 42)
    file_pattern = data_selection.get('file_pattern', '*.fz')  # Updated for DSI Studio .fz files
    
    # Set random seed for reproducible results
    if seed:
        random.seed(seed)
        logging.info(f"🎲 Random seed set to: {seed}")
    
    # Find available data files
    data_path = Path(source_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {source_dir}")
    
    # Get all files matching pattern
    all_files = list(data_path.glob(file_pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in {source_dir}")
    
    logging.info(f"📁 Found {len(all_files)} total files")
    
    # Handle 'all' selection method
    if count == "all" or count == len(all_files):
        count = len(all_files)
    
    # Select files based on method
    if method == 'random':
        if count >= len(all_files):
            selected_files = all_files
            logging.info(f"🎯 Selecting all {len(all_files)} available files")
        else:
            selected_files = random.sample(all_files, count)
            logging.info(f"🎯 Randomly selected {count} files")
    
    elif method == 'first':
        selected_files = sorted(all_files)[:count]
        logging.info(f"🎯 Selected first {len(selected_files)} files")
    
    elif method == 'specific':
        specific_files = data_selection.get('specific_subjects', [])
        selected_files = [data_path / f for f in specific_files if (data_path / f).exists()]
        logging.info(f"🎯 Selected {len(selected_files)} specific files")
    
    elif method == 'all':
        selected_files = all_files
        logging.info(f"🎯 Selecting all {len(all_files)} available files")
        
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    test_name = test_info.get('name', 'unnamed').replace(' ', '_').lower()
    count_str = "all" if count == len(all_files) else str(count)
    test_data_dir = Path(f"test_data_{test_name}_{method}_{count_str}")
    
    logging.info(f"📂 Test data directory: {test_data_dir}")
    
    # Create test data directory and copy files
    test_data_dir.mkdir(exist_ok=True)
    
    copied_files = []
    for file_path in selected_files:
        dest_path = test_data_dir / file_path.name
        if not dest_path.exists():
            shutil.copy2(file_path, dest_path)
            logging.debug(f"📋 Copied: {file_path.name}")
        copied_files.append(dest_path)
    
    logging.info(f"✅ Prepared {len(copied_files)} files in test directory")
    
    # Return configuration for pipeline
    pipeline_config = config.get('pipeline_config', {})
    
    return {
        'data_dir': str(test_data_dir),
        'steps_to_run': pipeline_config.get('steps_to_run', ['01', '02', '03', '04']),
        'output_base_dir': pipeline_config.get('output_base_dir', 'analysis_results'),
        'extraction_config': pipeline_config.get('extraction_config', 'dsi_studio_tools/research_config.json'),
        'test_info': test_info,
        'metadata': config.get('cross_validation_metadata', {})
    }

def load_test_configuration(test_config_path):
    """Load and process JSON test configuration."""
    
    # Optional: Validate configuration first
    try:
        from scripts.json_validator import validate_config_file
        if not validate_config_file(test_config_path):
            raise ValueError(f"Configuration validation failed: {test_config_path}")
    except ImportError:
        logging.warning("scripts/json_validator.py not available - skipping validation")
    
    with open(test_config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"🧪 Loading test configuration: {test_config_path}")
    
    return load_test_configuration_from_dict(config)
    
    if not all_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in {source_dir}")
    
    logging.info(f"📁 Found {len(all_files)} total files")
    
    # Handle "all" subjects option
    if count == "all" or count == len(all_files):
        count = len(all_files)
        logging.info(f"🔄 Processing ALL {count} subjects")
    
    # Select files based on method
    if method == 'random':
        if count >= len(all_files):
            selected_files = all_files
            logging.info(f"📝 Using all {len(selected_files)} files (requested >= available)")
        else:
            selected_files = random.sample(all_files, count)
            logging.info(f"📝 Randomly selected {len(selected_files)} files")
    elif method == 'first':
        selected_files = sorted(all_files)[:count]
        logging.info(f"📝 Selected first {len(selected_files)} files")
    elif method == 'specific':
        specific_files = data_selection.get('specific_subjects', [])
        selected_files = [data_path / f for f in specific_files if (data_path / f).exists()]
        logging.info(f"📝 Selected {len(selected_files)} specific files")
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    # Create temporary data directory with selected files
    test_name = test_info.get('name', 'test').replace(' ', '_').lower()
    count_str = "all" if count == len(all_files) else str(count)
    test_data_dir = Path(f"test_data_{test_name}_{method}_{count_str}")
    test_data_dir.mkdir(exist_ok=True)
    
    # Copy or symlink selected files
    for src_file in selected_files:
        dest_file = test_data_dir / src_file.name
        if not dest_file.exists():
            try:
                dest_file.symlink_to(src_file.absolute())
            except OSError:
                # Fallback to copying if symlink fails
                import shutil
                shutil.copy2(src_file, dest_file)
    
    logging.info(f"🔗 Created test data directory: {test_data_dir}")
    
    return config, test_data_dir

def run_bootstrap_qa_validation(data_dir, config, args):
    """
    Integrated bootstrap QA validation using the bootstrap_qa_validator.py.
    
    This runs automatically when bootstrap QA is enabled and performs:
    1. Create bootstrap configurations (20% in 2 waves)
    2. Run both bootstrap waves  
    3. Validate QA stability across waves
    4. Provide recommendation to proceed or adjust
    
    Returns True if QA validation passes, False otherwise.
    """
    logging.info("🔬 Starting integrated Bootstrap QA Validation...")
    
    try:
        # Step 1: Create bootstrap configurations
        logging.info("📋 Creating bootstrap QA configurations...")
        
        create_cmd = [
            "python", "scripts/bootstrap_qa_validator.py", "create", str(data_dir)
        ]
        
        if args.dry_run:
            logging.info(f"[DRY-RUN] Would run: {' '.join(create_cmd)}")
            result = type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
        else:
            result = subprocess.run(create_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"❌ Failed to create bootstrap configurations: {result.stderr}")
            return False
        
        logging.info("✅ Bootstrap configurations created successfully")
        
        # Step 2: Run bootstrap waves
        wave_configs = [
            "bootstrap_configs/bootstrap_qa_wave_1.json",
            "bootstrap_configs/bootstrap_qa_wave_2.json"
        ]
        
        bootstrap_results = []
        for wave_num, wave_config in enumerate(wave_configs, 1):
            if not Path(wave_config).exists():
                logging.error(f"❌ Bootstrap wave config not found: {wave_config}")
                return False
                
            logging.info(f"🌊 Running Bootstrap Wave {wave_num}...")
            
            # Run pipeline for this bootstrap wave
            wave_cmd = [
                "python", "run_pipeline.py", 
                "--test-config", wave_config
            ]
            
            if args.verbose:
                wave_cmd.append("--verbose")
                
            if args.dry_run:
                logging.info(f"[DRY-RUN] Would run: {' '.join(wave_cmd)}")
                result = type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
            else:
                result = subprocess.run(wave_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"❌ Bootstrap Wave {wave_num} failed: {result.stderr}")
                return False
                
            # Look for the bootstrap result directory
            wave_result_dir = f"bootstrap_results_bootstrap_qa_wave_{wave_num}"
            if Path(wave_result_dir).exists():
                bootstrap_results.append(wave_result_dir)
                
            logging.info(f"✅ Bootstrap Wave {wave_num} completed")
        
        if len(bootstrap_results) != 2:
            logging.error(f"❌ Expected 2 bootstrap result directories, found {len(bootstrap_results)}")
            return False
            
        # Step 3: Validate bootstrap results
        logging.info("📊 Validating bootstrap QA stability...")
        
        validate_cmd = [
            "python", "scripts/bootstrap_qa_validator.py", "validate"
        ] + bootstrap_results
        
        if args.dry_run:
            logging.info(f"[DRY-RUN] Would run: {' '.join(validate_cmd)}")
            result = type('R', (), {'returncode': 0, 'stdout': '{}', 'stderr': ''})()
        else:
            result = subprocess.run(validate_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"❌ Bootstrap QA validation failed: {result.stderr}")
            return False
            
        # Parse validation output to determine if it passed
        output = result.stdout
        try:
            # Try to parse JSON output first
            import json
            qa_data = json.loads(output)
            overall_rating = qa_data.get('overall_assessment', {}).get('overall_stability', 'UNKNOWN')
            score = qa_data.get('overall_assessment', {}).get('average_score', 0)
            
            if overall_rating in ['EXCELLENT', 'GOOD']:
                logging.info(f"✅ Bootstrap QA validation PASSED - Rating: {overall_rating} (Score: {score:.1f}/4.0)")
                return True
            elif overall_rating == 'FAIR':
                logging.warning(f"⚠️ Bootstrap QA validation shows fair stability - Rating: {overall_rating} (Score: {score:.1f}/4.0)")
                logging.warning("💡 Consider adjusting parameters or increasing sample size")
                return True  # Still proceed but with warnings
            else:
                logging.error(f"❌ Bootstrap QA validation failed - Rating: {overall_rating} (Score: {score:.1f}/4.0)")
                return False
                
        except json.JSONDecodeError:
            # Fallback to string parsing for older output format
            if "QA metrics are highly stable - proceed with full dataset analysis" in output:
                logging.info("✅ Bootstrap QA validation PASSED - proceeding with full dataset")
                return True
            elif "EXCELLENT" in output or "GOOD" in output:
                logging.info("✅ Bootstrap QA validation shows good stability - proceeding")
                return True
            else:
                logging.warning("⚠️ Bootstrap QA validation shows concerning stability")
                logging.warning("💡 Consider adjusting parameters or increasing sample size")
                return True  # Still proceed but with warnings
            
    except Exception as e:
        logging.error(f"❌ Bootstrap QA validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Braingraph Pipeline Runner - Steps 02-04",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline (steps 02-04)
    python run_pipeline.py --input results/organized_matrices/ --output analysis_results/
    
    # Run only optimization step
    python run_pipeline.py --input organized_matrices/ --step 02
    
    # Run with verbose logging
    python run_pipeline.py --input data/ --output results/ --verbose
    
    # Auto-detect input from step 01 output
    python run_pipeline.py --output analysis_results/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input directory (organized matrices from step 01). Will auto-detect if not specified.'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='analysis_results',
        help='Output directory for analysis results (default: analysis_results)'
    )
    
    parser.add_argument(
        '--step', '-s',
        choices=['01', '02', '03', '04', 'all', 'analysis'],
        default='all',
        help='Pipeline step to run: 01=extraction, 02=optimization, 03=selection, 04=statistics.\n'
             'all: runs 01-03. Statistical analysis is not included in this package.\n'
             'analysis: runs 02-04.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce console output (warnings/errors only)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file for pipeline parameters'
    )
    
    parser.add_argument(
        '--test-config', '-t',
        help='JSON test configuration file for automated testing with subset of subjects'
    )

    parser.add_argument(
        '--enable-bootstrap-qa',
        action='store_true',
        help='Enable automatic bootstrap QA validation (20%% in 2 waves) before full dataset processing (RECOMMENDED for production)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run: print actions without making changes or running external commands'
    )

    parser.add_argument(
        '--bootstrap-optimize',
        action='store_true',
        help='Run enhanced bootstrap parameter optimization workflow. Compare different parameter sets and choose optimal configuration for full analysis.'
    )

    # Optimizer tuning flags (forwarded to cross_validation_bootstrap_optimizer.py)
    parser.add_argument(
        '--optimizer-max-parallel',
        type=int,
        default=None,
        help='Max combinations to run in parallel per wave when using --bootstrap-optimize'
    )
    parser.add_argument(
        '--optimizer-prune-nonbest',
        action='store_true',
        help='Prune non-best combo outputs after selection when using --bootstrap-optimize'
    )

    parser.add_argument(
        '--cross-validated-config',
        help='JSON configuration file with cross-validated optimal parameters (output from cross_validation_bootstrap_optimizer.py)'
    )

    parser.add_argument(
    # Statistical analysis is out of scope for this package
    )

    # Extraction parameters (Step 01) - JSON Config Approach
    extraction_group = parser.add_argument_group('Step 01: Connectivity Extraction')
    
    extraction_group.add_argument(
        '--data-dir',
        help='Input data directory containing .fib.gz/.fz files (for step 01)'
    )
    
    extraction_group.add_argument(
        '--extraction-config',
        help='JSON configuration file for extraction parameters (like sweep_config.json). '
             'Replaces individual parameter flags. See example configs.'
    )
    
    extraction_group.add_argument(
        '--pilot', action='store_true',
        help='Run pilot test on subset of files'
    )
    
    extraction_group.add_argument(
        '--pilot-count', type=int, default=2,
        help='Number of files for pilot test (default: 2)'
    )
    
    extraction_group.add_argument(
        '--batch', action='store_true',
        help='Batch mode: process all files in data directory'
    )
    
    args = parser.parse_args()

    # If invoked with no CLI args, print help/usage to satisfy project conventions
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Check if no meaningful arguments provided - show help
    if (not args.test_config and not args.cross_validated_config and not args.data_dir and not args.input and 
        args.step in ['01', 'all'] and not args.config):
        print("🧠 BRAINGRAPH PIPELINE - BRAIN CONNECTIVITY ANALYSIS")
        print("=" * 60)
        print()
        print("🚀 QUICK START (RECOMMENDED):")
        print()
        print("  # Test pipeline with 5 subjects")
        print("  python run_pipeline.py --test-config test_full_pipeline.json")
        print()
        print("  # Production run with all subjects")
        print("  python run_pipeline.py --test-config test_all_subjects.json")
        print()
        print("📋 AVAILABLE TEST CONFIGURATIONS:")
        test_configs = [
            ("test_full_pipeline.json", "Complete 4-step test (5 subjects)"),
            ("test_all_subjects.json", "Production run (all subjects)"),
            ("test_extraction_only.json", "Test only Step 01 (3 subjects)")
        ]
        
        for config, desc in test_configs:
            if Path(config).exists():
                print(f"  ✅ {config:<25} - {desc}")
            else:
                print(f"  ❌ {config:<25} - {desc} (missing)")
        print()
        print("🔧 MANUAL USAGE:")
        print()
        print("  # Individual steps")
        print("  python run_pipeline.py --step 01 --data-dir /path/to/data --extraction-config optimal_config.json")
        print("  python run_pipeline.py --step 02 --input organized_matrices/")
        print("  python run_pipeline.py --step 03")
        print("  python run_pipeline.py --step 04")
        print()
        print("🛠️ SETUP & VALIDATION:")
        print()
        print("  # Install environment")
        print("  ./00_install_new.sh")
        print("  source braingraph_pipeline/bin/activate")
        print()
        print("  # Validate setup")
        print("  python scripts/validate_setup.py --config 01_working_config.json")
        print()
        print("  # Validate JSON configuration")
        print("  python scripts/json_validator.py your_config.json")
        print()
        print("📖 For detailed help:")
        print("  python run_pipeline.py --help")
        print("  See README.md for comprehensive documentation")
        print()
        print("🎯 TIP: Start with 'python run_pipeline.py --test-config test_full_pipeline.json'")
        print("=" * 60)
        sys.exit(0)
    
    # Determine output base directory early for logging placement
    intended_output_dir = args.output if hasattr(args, 'output') and args.output else 'analysis_results'

    # Set up logging (log file goes into the output directory)
    logger = setup_logging(args.verbose, args.quiet, intended_output_dir)
    
    # Handle cross-validated configuration mode
    test_config = None
    test_data_dir = None
    if args.cross_validated_config:
        try:
            config_dict = load_cross_validated_configuration(args.cross_validated_config)
            
            # Override pipeline parameters from cross-validated config
            args.data_dir = config_dict['data_dir']
            args.extraction_config = config_dict['extraction_config']
            args.output_base_dir = config_dict['output_base_dir']
            
            # Set steps to run from config
            if args.step == 'all':
                steps_list = config_dict['steps_to_run']
                if len(steps_list) == 1:
                    args.step = steps_list[0]
            
            logging.info(f"🔬 Running with cross-validated parameters")
            logging.info(f"📊 Validation metadata: {config_dict.get('metadata', {})}")
            
        except Exception as e:
            logging.error(f"❌ Failed to load cross-validated configuration: {e}")
            return 1
    
    # Handle test configuration mode
    elif args.test_config:
        try:
            test_config, test_data_dir = load_test_configuration(args.test_config)
            
            # Override pipeline parameters from test config
            pipeline_config = test_config.get('pipeline_config', {})
            
            # Set extraction config if specified in test
            if 'extraction_config' in pipeline_config and not args.extraction_config:
                args.extraction_config = pipeline_config['extraction_config']
            
            # Set steps to run if specified in test
            if 'steps_to_run' in pipeline_config and args.step == 'all':
                steps_list = pipeline_config['steps_to_run']
                if len(steps_list) == 1:
                    args.step = steps_list[0]
                elif set(steps_list) == {'02', '03', '04'}:
                    args.step = 'analysis'
                # Keep 'all' for complete pipeline
            
            # Override output directory if specified
            if 'output_base_dir' in pipeline_config:
                args.output = pipeline_config['output_base_dir']
            
            # Override data directory for step 01
            if not args.data_dir and test_data_dir:
                args.data_dir = str(test_data_dir)
                
            logger.info(f"🧪 Test mode enabled - using {len(list(test_data_dir.glob('*')))} subjects")
            
        except Exception as e:
            logger.error(f"❌ Failed to load test configuration: {e}")
            sys.exit(1)
    
    # Bootstrap QA Validation (if enabled)
    if args.enable_bootstrap_qa and args.test_config:
        # Check if this is a production dataset that should have bootstrap QA
        config_name = Path(args.test_config).stem.lower()
        if 'all_subjects' in config_name or 'production' in config_name:
            logger.info("🔬 Bootstrap QA validation enabled for production dataset")
            
            # Use the original data directory from the test config, not the test_data symlink dir
            original_data_dir = test_config.get('data_selection', {}).get('source_dir')
            if not original_data_dir:
                logger.error("❌ Cannot run bootstrap QA: no source_dir in test configuration")
                sys.exit(1)
                
            # Run bootstrap QA validation using original data directory
            qa_passed = run_bootstrap_qa_validation(original_data_dir, test_config, args)
            if not qa_passed:
                logger.error("❌ Bootstrap QA validation failed - stopping pipeline")
                logger.info("💡 Consider adjusting parameters or manually reviewing results")
                sys.exit(1)
                
            logger.info("✅ Bootstrap QA validation passed - continuing with full dataset")
        else:
            logger.info("ℹ️  Bootstrap QA skipped (not a production dataset)")
    elif args.enable_bootstrap_qa:
        logger.warning("⚠️  Bootstrap QA enabled but no test config provided - skipping")

    # Bootstrap Parameter Optimization Workflow
    if args.bootstrap_optimize:
        logger.info("🔬 Starting Bootstrap Parameter Optimization Workflow")
        
        # Determine data directory for optimization
        if args.data_dir:
            optimization_data_dir = args.data_dir
        elif test_config:
            optimization_data_dir = test_config.get('data_selection', {}).get('source_dir')
        else:
            logger.error("❌ Bootstrap optimization requires --data-dir or --test-config")
            sys.exit(1)
            
        if not optimization_data_dir or not Path(optimization_data_dir).exists():
            logger.error(f"❌ Data directory not found: {optimization_data_dir}")
            sys.exit(1)
            
        # Run cross-validation bootstrap parameter optimization
        optimize_cmd = [
            "python", "scripts/cross_validation_bootstrap_optimizer.py",
            "--data-dir", optimization_data_dir,
            "--output-dir", args.output
        ]

        # Forward optimizer tuning flags if provided
        if args.optimizer_max_parallel and args.optimizer_max_parallel > 1:
            optimize_cmd += ["--max-parallel", str(args.optimizer_max_parallel)]
        if args.optimizer_prune_nonbest:
            optimize_cmd += ["--prune-nonbest"]
        
        logger.info("🚀 Launching bootstrap parameter optimizer...")
        logger.info(f"   Data directory: {optimization_data_dir}")
        logger.info(f"   Output directory: {args.output}")
        
        # Forward an extraction-config if the user provided one; optimizer will attach it to generated waves
        if args.extraction_config:
            optimize_cmd += ["--extraction-config", args.extraction_config]

        if args.dry_run:
            logger.info(f"[DRY-RUN] Would run: {' '.join(optimize_cmd)}")
            result = type('R', (), {'returncode': 0})()
        else:
            result = subprocess.run(optimize_cmd)

        if result.returncode == 0:
            logger.info("✅ Bootstrap parameter optimization completed successfully!")
            logger.info("💡 Check the generated optimized_full_analysis_config.json for next steps")
        else:
            logger.error("❌ Bootstrap parameter optimization failed")
            
        # Exit after optimization - user needs to run full pipeline separately
        sys.exit(result.returncode)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🧠 Braingraph Pipeline Runner")
    logger.info("=" * 50)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Steps to run: {args.step}")
    
    # Find input directory or data directory
    if args.step == '01' or args.step == 'all':
        # For step 01, we need raw data directory
        if args.data_dir:
            data_path = Path(args.data_dir)
            if not data_path.exists():
                logger.error(f"Data directory does not exist: {data_path}")
                sys.exit(1)
            logger.info(f"📁 Data directory: {data_path}")
        elif args.input:
            data_path = Path(args.input)
            if not data_path.exists():
                logger.error(f"Input directory does not exist: {data_path}")
                sys.exit(1)
            logger.info(f"📁 Data directory: {data_path}")
        else:
            logger.error("For step 01, please specify --data-dir or --input with raw data directory")
            sys.exit(1)
    
    # For steps 02-04, find organized matrices
    if args.step in ['02', '03', '04', 'analysis'] or (args.step == 'all' and args.input):
        if args.input:
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"Input directory does not exist: {input_path}")
                sys.exit(1)
        else:
            logger.info("🔍 Auto-detecting input directory from step 01 output...")
            input_path = find_organized_matrices(Path.cwd())
            if not input_path:
                logger.error("Could not auto-detect input directory. Please specify --input")
                logger.error("Looking for: organized_matrices/ or directories with atlas structure")
                sys.exit(1)
            logger.info(f"📁 Auto-detected input: {input_path}")
    else:
        # Set input_path to None for steps that don't need it
        input_path = None
    
    # Define pipeline steps
    steps = {
        '01': {
            'script': 'scripts/extract_connectivity_matrices.py',
            'name': 'Connectivity Extraction',
            'args': []  # Will be populated based on args
        }
    }
    
    # Add steps that require input_path only if it's available OR if we're running the full pipeline
    if input_path or args.step == 'all':
        # For full pipeline, set input path to the output of step 01
        if args.step == 'all' and not input_path:
            pipeline_input_path = output_path / 'organized_matrices'
        else:
            pipeline_input_path = input_path
            
        steps.update({
            '02': {
                'script': 'scripts/metric_optimizer.py',
                'name': 'Network Quality Optimization',
                'args': [str(output_path / 'aggregated_network_measures.csv'), str(output_path / 'optimization_results')]
            },
            '03': {
                'script': 'scripts/optimal_selection.py', 
                'name': 'Quality-Based Selection',
                'args': [str(output_path / 'optimization_results' / 'optimized_metrics.csv'), 
                        str(output_path / 'selected_combinations')]
            }
        })
    
    # Configure step 01 arguments if needed
    if args.step == '01' or args.step == 'all':
        # Start with positional arguments: input and output
        step01_args = [str(data_path), str(output_path / 'organized_matrices')]
        
        # Add optional arguments
        # Use JSON config approach if provided
        if args.extraction_config:
            step01_args.extend(['--config', args.extraction_config])
        elif args.config:
            # Fallback to main config if no extraction-specific config
            step01_args.extend(['--config', args.config])
        
        # Add pilot mode if requested
        if args.pilot:
            step01_args.append('--pilot')
            if args.pilot_count:
                step01_args.extend(['--pilot-count', str(args.pilot_count)])
        
        # Add batch mode for directory processing
        step01_args.append('--batch')

        # Pass verbosity flags to extraction
        # Quiet is on by default in extraction; only pass debug when verbose
        if args.verbose:
            # When the pipeline is verbose, include DSI command details
            step01_args.append('--debug-dsi')
        
        steps['01']['args'] = step01_args

    # Add config to all steps if provided
    if args.config:
        for step_info in steps.values():
            step_info['args'].extend(['--config', args.config])
    
    # Determine which steps to run
    if args.step == 'all':
        # Full run covers steps 01–03 only (statistics out of scope)
        steps_to_run = ['01', '02', '03']
    elif args.step == 'analysis':
        # Analysis shorthand runs steps 02–03 only
        steps_to_run = ['02', '03']
    else:
        steps_to_run = [args.step]
    
    # Run the pipeline steps
    success_count = 0
    total_steps = len(steps_to_run)
    
    for step_num in steps_to_run:
        step_info = steps[step_num]
        
        logger.info(f"\n📊 Step {step_num}: {step_info['name']}")
        logger.info("-" * 50)
        
        # Create required directories for each step
        if step_num == '02':
            os.makedirs(output_path / 'optimization_results', exist_ok=True)
            # Aggregate network measures before running metric optimizer
            logger.info("📊 Aggregating network measures for optimization...")
            try:
                from scripts.aggregate_network_measures import aggregate_network_measures
                aggregate_success = aggregate_network_measures(
                    str(output_path / 'organized_matrices'),
                    str(output_path / 'aggregated_network_measures.csv')
                )
                if not aggregate_success:
                    logger.error("❌ Failed to aggregate network measures")
                    success = False
                    break
                logger.info("✅ Network measures aggregated successfully")
            except Exception as e:
                logger.error(f"❌ Error during aggregation: {e}")
                success = False
                break
        elif step_num == '03':
            os.makedirs(output_path / 'selected_combinations', exist_ok=True)
        elif step_num == '04':
            os.makedirs(output_path / 'statistical_results', exist_ok=True)
        
        success = run_step(
            step_info['script'], 
            step_info['args'], 
            logger, 
            f"Step {step_num}",
            quiet=args.quiet
        )
        
        if success:
            success_count += 1
        else:
            logger.error(f"Pipeline failed at step {step_num}")
            if args.step == 'all':
                logger.error("Stopping pipeline due to failure")
                break
            else:
                sys.exit(1)
    
    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 Pipeline Summary")
    logger.info("=" * 50)
    logger.info(f"Steps completed: {success_count}/{total_steps}")
    logger.info(f"Output directory: {output_path.absolute()}")
    
    if success_count == total_steps:
        logger.info("✅ Pipeline completed successfully!")
        
        # Show next steps
        logger.info("\n📋 Results Available:")
        if '02' in steps_to_run:
            logger.info(f"  📊 Optimization results: {output_path}/optimization_results/")
        if '03' in steps_to_run:
            logger.info(f"  🎯 Selected combinations: {output_path}/selected_combinations/")
        if '04' in steps_to_run:
            logger.info(f"  📈 Statistical analysis: {output_path}/statistical_results/")
        
        # Run quality checks if in test mode
        if test_config:
            logger.info("\n" + "🔍" * 50)
            logger.info("🧪 Running Test Quality Checks")
            logger.info("🔍" * 50)
            
            try:
                quality_config = test_config.get('quality_checks', {})
                if quality_config.get('run_uniqueness_check', True):
                    # Import the quick quality check function
                    try:
                        from quick_quality_check import quick_uniqueness_check, quality_outlier_analysis
                        
                        # Run parameter uniqueness check
                        logger.info("📊 Checking parameter uniqueness...")
                        uniqueness_result = quick_uniqueness_check(str(output_path))
                        
                        if uniqueness_result:
                            logger.info("✅ Parameter uniqueness check passed")
                            logger.info(f"   📈 Diversity ranges: {uniqueness_result}")
                        else:
                            logger.warning("⚠️  Parameter uniqueness check failed or no data")
                        
                        # Run quality outlier analysis
                        if quality_config.get('run_outlier_analysis', False):
                            logger.info("🎯 Checking for quality outliers...")
                            outlier_result = quality_outlier_analysis(str(output_path))
                            
                            if outlier_result:
                                logger.info("✅ Quality outlier analysis completed")
                            else:
                                logger.warning("⚠️  Quality outlier analysis failed or no data")
                        
                    except ImportError:
                        logger.warning("⚠️  Quality check module 'quick_quality_check.py' not found")
                
                # Cleanup test data directory if requested
                cleanup_config = test_config.get('cleanup', {})
                if cleanup_config.get('remove_intermediate_files', False) and test_data_dir:
                    import shutil
                    try:
                        shutil.rmtree(test_data_dir)
                        logger.info(f"🧹 Cleaned up test data directory: {test_data_dir}")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to cleanup test directory: {e}")
                
            except Exception as e:
                logger.error(f"❌ Quality checks failed: {e}")
            
    else:
        logger.error("❌ Pipeline completed with errors")
        sys.exit(1)

if __name__ == '__main__':
    main()
