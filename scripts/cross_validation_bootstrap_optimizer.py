#!/usr/bin/env python3
"""
Cross-Validation Bootstrap Optimizer
===================================

Runs parameter optimization across two bootstrap waves to find
optimal DSI Studio parameters using cross-validation.

Author: Braingraph Pipeline Team
"""

import json
import os
import sys
import subprocess
import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.utils.runtime import configure_stdio
from scripts.sweep_utils import (
    build_param_grid_from_config,
    grid_product,
    random_sampling as sweep_random_sampling,
    lhs_sampling,
    apply_param_choice_to_config,
)

def setup_logging(output_dir: str | None = None):
    """Set up logging configuration.

    Writes a cross_validation_*.log file into the output directory if provided; otherwise, logs only to console.
    """
    # Console without timestamps; file with timestamps
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    handlers = [console_handler]
    if output_dir:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            file_handler = logging.FileHandler(
                str(Path(output_dir) / f'cross_validation_{timestamp}.log'),
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            handlers.append(file_handler)
        except Exception:
            # Fallback to console-only if cannot create file handler
            pass
    logging.basicConfig(level=logging.INFO, handlers=handlers)

def repo_root() -> Path:
    """Return the repository root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def generate_wave_configs(data_dir, output_dir, n_subjects: int = 3, extraction_cfg: str | None = None):
    """Generate wave configuration files.

    Parameters
    ----------
    data_dir: str | Path
        Source directory containing subject files.
    output_dir: str | Path
        Base output directory for wave configs.
    n_subjects: int
        Number of subjects to sample per wave.
    """
    
    # Create configs subdirectory
    configs_dir = Path(output_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Wave 1 configuration
    # Choose extraction config path
    if not extraction_cfg:
        extraction_cfg = "configs/braingraph_default_config.json"

    wave1_config = {
        "test_config": {
            "name": "bootstrap_qa_wave_1",
            "description": "Bootstrap QA Wave 1 - Quick test validation"
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": int(n_subjects),
            "random_seed": 42,
            "file_pattern": "*.fz"
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": extraction_cfg
        },
        "bootstrap": {
            "n_iterations": 5,
            "sample_ratio": 0.8
        }
    }
    
    # Wave 2 configuration (different random seed)
    wave2_config = {
        "test_config": {
            "name": "bootstrap_qa_wave_2",
            "description": "Bootstrap QA Wave 2 - Cross-validation"
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": int(n_subjects),
            "random_seed": 1337,  # Different seed for different subject sample
            "file_pattern": "*.fz"
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": extraction_cfg
        },
        "bootstrap": {
            "n_iterations": 5,
            "sample_ratio": 0.8
        }
    }
    
    # Save configurations
    wave1_path = configs_dir / "wave1_config.json"
    wave2_path = configs_dir / "wave2_config.json"
    
    with open(wave1_path, 'w') as f:
        json.dump(wave1_config, f, indent=2)
    with open(wave2_path, 'w') as f:
        json.dump(wave2_config, f, indent=2)
        
    logging.info(f"📝 Generated wave configurations in {configs_dir}")
    
    return str(wave1_path), str(wave2_path)


def generate_single_wave_config(data_dir, output_dir, n_subjects: int = 5, extraction_cfg: str | None = None):
    """Generate single wave configuration for comprehensive optimization.

    Parameters
    ----------
    data_dir: str | Path
        Source directory containing subject files.
    output_dir: str | Path
        Base output directory for wave config.
    n_subjects: int
        Number of subjects to sample (can be more than default 3).
    """
    
    # Create configs subdirectory
    configs_dir = Path(output_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Single comprehensive wave configuration
    if not extraction_cfg:
        extraction_cfg = "configs/braingraph_default_config.json"

    wave_config = {
        "test_config": {
            "name": "comprehensive_optimization",
            "description": "Single comprehensive wave for parameter optimization"
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": int(n_subjects),
            "random_seed": 42,
            "file_pattern": "*.fz"
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": extraction_cfg
        },
        "bootstrap": {
            "n_iterations": 5,
            "sample_ratio": 0.8
        }
    }
    
    # Save configuration
    wave_path = configs_dir / "comprehensive_wave.json"
    
    with open(wave_path, 'w') as f:
        json.dump(wave_config, f, indent=2)
    
    logging.info(f"📝 Generated single comprehensive wave configuration in {configs_dir}")
    
    return str(wave_path)

def load_wave_config(config_file):
    """Load wave configuration."""
    with open(config_file, 'r') as f:
        return json.load(f)

def run_wave_pipeline(wave_config_file, output_base_dir, max_parallel: int = 1, prune_nonbest: bool = False, verbose: bool = False):
    """Run pipeline for a single wave."""
    logging.info(f"🚀 Running pipeline for {wave_config_file}")
    
    # Load wave configuration
    wave_config = load_wave_config(wave_config_file)
    wave_name = wave_config['test_config']['name']
    
    logging.info(f"📋 Wave configuration loaded:")
    logging.info(f"   • Name: {wave_name}")
    logging.info(f"   • Data source: {wave_config['data_selection']['source_dir']}")
    logging.info(f"   • Subset size (n_subjects): {wave_config['data_selection'].get('n_subjects')}")
    logging.info(f"   • Bootstrap iterations: {wave_config.get('bootstrap', {}).get('n_iterations')}")
    
    # Create output directory for this wave
    wave_output_dir = Path(output_base_dir) / wave_name
    wave_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Created wave output directory: {wave_output_dir}")

    # List all available files in source and save manifest
    try:
        src_dir = Path(wave_config['data_selection']['source_dir'])
        patterns = [wave_config['data_selection'].get('file_pattern', '*.fz')]
        files = []
        for pat in patterns:
            files.extend(sorted([p for p in src_dir.rglob(pat)]))
        # Also include .fib.gz if not already covered
        files.extend(sorted([p for p in src_dir.rglob('*.fib.gz')]))
        # Deduplicate
        seen = set()
        uniq = []
        for p in files:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        available_manifest = wave_output_dir / 'available_files.txt'
        with available_manifest.open('w') as mf:
            for p in uniq:
                mf.write(str(p) + "\n")
        logging.info(f"📄 Available files listed: {available_manifest} ({len(uniq)})")
    except Exception as e:
        logging.warning(f"⚠️  Could not list available files: {e}")
    
    # Determine selection for this wave
    n_subjects = int(wave_config['data_selection'].get('n_subjects') or 3)
    seed = int(wave_config['data_selection'].get('random_seed') or 42)
    random.seed(seed)
    # Prefer .fz, then .fib.gz
    fz_files = [p for p in uniq if str(p).endswith('.fz')]
    fib_files = [p for p in uniq if str(p).endswith('.fib.gz')]
    pool = fz_files + fib_files
    if not pool:
        logging.error("❌ No candidate files found for selection")
        return False
    if n_subjects >= len(pool):
        selected = pool
    else:
        selected = random.sample(pool, n_subjects)
    # Write selected manifest and build staging dir with symlinks
    selected_manifest = wave_output_dir / 'selected_files.txt'
    with selected_manifest.open('w') as sf:
        for p in selected:
            sf.write(str(p) + "\n")
    logging.info(f"📄 Selected files listed: {selected_manifest} ({len(selected)})")
    staging_dir = wave_output_dir / 'selected_data'
    staging_dir.mkdir(exist_ok=True)
    for p in selected:
        dest = staging_dir / p.name
        try:
            if not dest.exists():
                dest.symlink_to(p)
        except OSError:
            # Fallback to copy if symlink not permitted
            shutil.copy2(p, dest)
    logging.info(f"📂 Staging data directory: {staging_dir}")

    # Run the pipeline for this wave
    root = repo_root()
    extraction_cfg_rel = wave_config.get('pipeline_config', {}).get('extraction_config', 'configs/braingraph_default_config.json')
    extraction_cfg = str((root / extraction_cfg_rel).resolve()) if not Path(extraction_cfg_rel).is_absolute() else extraction_cfg_rel
    logging.info(f"🔧 Wave '{wave_name}' using extraction config: {extraction_cfg}")
    
    # Load base extraction config to determine sweep combinations
    try:
        with open(extraction_cfg, 'r') as _f:
            base_cfg = json.load(_f)
    except Exception as e:
        logging.error(f"❌ Failed to load extraction config {extraction_cfg}: {e}")
        return False

    sp = (base_cfg.get('sweep_parameters') or {})
    param_values, mapping = build_param_grid_from_config({'sweep_parameters': sp})
    # Determine sampling strategy (default: grid over all values)
    sampling = (sp.get('sampling') or {}) if isinstance(sp, dict) else {}
    method = (sampling.get('method') or 'grid').lower()
    n_samples = int(sampling.get('n_samples') or 0)
    seed = int(sampling.get('random_seed') or 42)
    if method == 'grid' or not param_values:
        combos = grid_product(param_values) if param_values else [{}]
    elif method == 'random':
        if n_samples <= 0:
            n_samples = 24
        combos = sweep_random_sampling(param_values, n_samples, seed)
    else:  # lhs
        if n_samples <= 0:
            n_samples = 24
        combos = lhs_sampling(param_values, n_samples, seed)

    # Prepare sweep directories
    sweep_cfg_dir = wave_output_dir / 'configs' / 'sweep'
    sweep_cfg_dir.mkdir(parents=True, exist_ok=True)
    combos_dir = wave_output_dir / 'combos'
    combos_dir.mkdir(parents=True, exist_ok=True)

    # Helper to echo parameters consistently
    preferred_order = [
        'tract_count', 'connectivity_threshold', 'otsu_threshold', 'fa_threshold',
        'min_length', 'max_length', 'track_voxel_ratio', 'turning_angle', 'step_size',
        'smoothing', 'dt_threshold',
    ]
    def fmt_choice(c: dict) -> str:
        items = []
        used = set()
        for k in preferred_order:
            if k in c:
                items.append(f"{k}={c[k]}")
                used.add(k)
        for k in sorted([k for k in c.keys() if k not in used]):
            items.append(f"{k}={c[k]}")
        return ', '.join(items)

    # Execute Step 01+02 for each combination
    optimized_csvs = []
    logging.info(f"⏳ Starting parameter sweep for {wave_name}: {len(combos)} combination(s) [method={method}, max_parallel={max_parallel}]")

    base_threads = int(base_cfg.get('thread_count') or 8)
    adj_threads = max(1, base_threads // max(1, int(max_parallel)))

    # Prepare tasks
    tasks = []
    for i, choice in enumerate(combos, 1):
        # Build derived config with thread_count scaling
        derived = apply_param_choice_to_config(base_cfg, choice, mapping)
        try:
            import datetime as _dt
            derived['sweep_meta'] = {
                'index': i,
                'choice': choice,
                'sampler': method,
                'total_combinations': len(combos),
                'source_config': extraction_cfg,
                'generated_at': _dt.datetime.now().isoformat(timespec='seconds'),
            }
        except Exception:
            pass
        derived['thread_count'] = adj_threads

        cfg_path = sweep_cfg_dir / f'sweep_{i:04d}.json'
        with cfg_path.open('w') as _out:
            json.dump(derived, _out, indent=2)

        combo_out = combos_dir / f'sweep_{i:04d}'
        combo_out.mkdir(parents=True, exist_ok=True)

        # Echo parameters
        logging.info(f"🔎 Parameters [{i}/{len(combos)}]: {fmt_choice(choice)} | thread_count={adj_threads}")

        tasks.append((i, cfg_path, combo_out, False))

    def run_combo(i: int, cfg_path: Path, combo_out: Path, verbose: bool = False) -> tuple[Path, Path, float, int, str, str]:
        """Run step01+aggregate+step02 for a single combination.
        Returns (cfg_path, optimized_csv_path, selection_score, tract_count, status)"""
        env = os.environ.copy()
        env.setdefault('PYTHONUNBUFFERED', '1')
        # Step 01
        cmd01 = [
            sys.executable, '-u', str(root / 'scripts' / 'run_pipeline.py'),
            '--data-dir', str(staging_dir), '--step', '01', '--output', str(combo_out),
            '--extraction-config', str(cfg_path),
        ]
        p1 = subprocess.run(cmd01, capture_output=True, text=True, env=env)
        # If verbose, print DSI Studio command from step01 output
        if verbose and p1.stdout:
            for line in p1.stdout.splitlines():
                if "DSI Studio command:" in line:
                    logging.info(f"[VERBOSE] {line}")
        if p1.returncode != 0:
            # Persist failure diagnostics for this combo
            try:
                fail_diag = {
                    'status': 'failed',
                    'stage': 'step01',
                    'wave': wave_name,
                    'combo_dir': str(combo_out),
                    'config_path': str(cfg_path),
                    'return_code': p1.returncode,
                    'stdout_tail': p1.stdout[-4000:] if p1.stdout else '',
                    'stderr_tail': p1.stderr[-4000:] if p1.stderr else '',
                }
                (combo_out / 'diagnostics.json').write_text(json.dumps(fail_diag, indent=2))
            except Exception:
                pass
            return (cfg_path, Path(''), -1.0, -1, f"step01_failed: rc={p1.returncode}\n{p1.stdout[-4000:]}\n{p1.stderr[-4000:] if p1.stderr else ''}")

        # Aggregate measures
        agg_csv = combo_out / '01_connectivity' / 'aggregated_network_measures.csv'
        if not agg_csv.exists():
            cmdAgg = [
                sys.executable, str(root / 'scripts' / 'aggregate_network_measures.py'),
                str(combo_out / '01_connectivity'), str(agg_csv)
            ]
            pAgg = subprocess.run(cmdAgg, capture_output=True, text=True)
            if pAgg.returncode != 0 or not agg_csv.exists():
                # Persist failure diagnostics for this combo
                try:
                    fail_diag = {
                        'status': 'failed',
                        'stage': 'aggregate',
                        'wave': wave_name,
                        'combo_dir': str(combo_out),
                        'config_path': str(cfg_path),
                        'return_code': pAgg.returncode,
                        'stdout_tail': pAgg.stdout[-4000:] if pAgg.stdout else '',
                        'stderr_tail': pAgg.stderr[-4000:] if pAgg.stderr else '',
                    }
                    (combo_out / 'diagnostics.json').write_text(json.dumps(fail_diag, indent=2))
                except Exception:
                    pass
                return (cfg_path, Path(''), -1.0, -1, f"aggregate_failed: rc={pAgg.returncode}\n{pAgg.stdout[-4000:]}\n{pAgg.stderr[-4000:] if pAgg.stderr else ''}", '')

        # Step 02
        step02_dir = combo_out / '02_optimization'
        step02_dir.mkdir(exist_ok=True)
        cmd02 = [
            sys.executable, str(root / 'scripts' / 'metric_optimizer.py'),
            '-i', str(agg_csv), '-o', str(step02_dir)
        ]
        p2 = subprocess.run(cmd02, capture_output=True, text=True)
        opt_csv = step02_dir / 'optimized_metrics.csv'
        if p2.returncode != 0 or not opt_csv.exists():
            # Persist failure diagnostics for this combo
            try:
                fail_diag = {
                    'status': 'failed',
                    'stage': 'step02',
                    'wave': wave_name,
                    'combo_dir': str(combo_out),
                    'config_path': str(cfg_path),
                    'return_code': p2.returncode,
                    'stdout_tail': p2.stdout[-4000:] if p2.stdout else '',
                    'stderr_tail': p2.stderr[-4000:] if p2.stderr else '',
                }
                (combo_out / 'diagnostics.json').write_text(json.dumps(fail_diag, indent=2))
            except Exception:
                pass
            return (cfg_path, Path(''), -1.0, -1, f"step02_failed: rc={p2.returncode}\n{p2.stdout[-4000:]}\n{p2.stderr[-4000:] if p2.stderr else ''}")
        # Evaluate score
        try:
            df = pd.read_csv(opt_csv)
            # Use absolute (raw) mean as primary selector to avoid trivial 1.0 normalization
            raw_mean = float(df['quality_score_raw'].mean()) if 'quality_score_raw' in df.columns else float('nan')
            norm_max = float(df['quality_score'].max()) if 'quality_score' in df.columns else float('nan')
            score = raw_mean if not np.isnan(raw_mean) else (norm_max if not np.isnan(norm_max) else -1.0)
            # Extract tract_count and sweep meta from cfg for tie-breakers and reporting
            try:
                with open(cfg_path, 'r') as _cf:
                    _cfg_json = json.load(_cf)
                tract_count = int(_cfg_json.get('sweep_parameters', {}).get('tract_count', _cfg_json.get('tract_count', -1)))
                thread_count = int(_cfg_json.get('thread_count') or -1)
                sweep_meta = _cfg_json.get('sweep_meta') or {}
            except Exception:
                tract_count = -1
                thread_count = -1
                sweep_meta = {}
            # Diagnostics from aggregated measures
            dens = float('nan')
            geff = float('nan')
            sw_b = float('nan')
            sw_w = float('nan')
            try:
                agg_csv = combo_out / '01_connectivity' / 'aggregated_network_measures.csv'
                diag_df = pd.read_csv(agg_csv)
                dens = float(diag_df['density'].mean()) if 'density' in diag_df.columns else float('nan')
                geff = float(diag_df['global_efficiency(weighted)'].mean()) if 'global_efficiency(weighted)' in diag_df.columns else float('nan')
                sw_b = float(diag_df['small-worldness(binary)'].mean()) if 'small-worldness(binary)' in diag_df.columns else float('nan')
                sw_w = float(diag_df['small-worldness(weighted)'].mean()) if 'small-worldness(weighted)' in diag_df.columns else float('nan')
            except Exception:
                pass

            # Persist per-combo diagnostics JSON
            try:
                diag_json = {
                    'status': 'ok',
                    'wave': wave_name,
                    'combo_dir': str(combo_out),
                    'config_path': str(cfg_path),
                    'combo_index': int(sweep_meta.get('index') or i),
                    'total_combinations': int(sweep_meta.get('total_combinations') or -1),
                    'sampler': sweep_meta.get('sampler'),
                    'parameters': sweep_meta.get('choice'),
                    'thread_count': thread_count,
                    'tract_count': tract_count,
                    'selection_score': float(score),
                    'quality_score_raw_mean': float(raw_mean) if not np.isnan(raw_mean) else None,
                    'quality_score_norm_max': float(norm_max) if not np.isnan(norm_max) else None,
                    'aggregates': {
                        'density_mean': None if np.isnan(dens) else float(dens),
                        'global_efficiency_weighted_mean': None if np.isnan(geff) else float(geff),
                        'small_worldness_binary_mean': None if np.isnan(sw_b) else float(sw_b),
                        'small_worldness_weighted_mean': None if np.isnan(sw_w) else float(sw_w),
                    },
                    'files': {
                        'optimized_metrics_csv': str(opt_csv),
                        'aggregated_measures_csv': str(agg_csv),
                    },
                }
                (combo_out / 'diagnostics.json').write_text(json.dumps(diag_json, indent=2))
            except Exception:
                pass

            # Human-readable diag string for logs
            extra_bits = []
            if not np.isnan(dens):
                extra_bits.append(f"density_mean={dens:.4f}")
            if not np.isnan(geff):
                extra_bits.append(f"geff_w_mean={geff:.4f}")
            diag = ' '.join(extra_bits)
        except Exception as e:
            # Persist failure cause if scoring failed
            try:
                fail_diag = {
                    'status': 'failed',
                    'stage': 'score',
                    'wave': wave_name,
                    'combo_dir': str(combo_out),
                    'config_path': str(cfg_path),
                    'error': str(e),
                }
                (combo_out / 'diagnostics.json').write_text(json.dumps(fail_diag, indent=2))
            except Exception:
                pass
            return (cfg_path, opt_csv, -1.0, -1, f"score_error: {e}", '')
        return (cfg_path, opt_csv, score, tract_count, 'ok', diag)

    optimized_csvs = []
    if max_parallel <= 1:
        for i, cfg_path, combo_out, verbose_flag in tasks:
            cfg, opt_csv, score, tc, status, diag = run_combo(i, cfg_path, combo_out, verbose_flag)
            if status == 'ok':
                try:
                    df = pd.read_csv(opt_csv)
                    raw_mean = float(df['quality_score_raw'].mean()) if 'quality_score_raw' in df.columns else float('nan')
                    norm_max = float(df['quality_score'].max()) if 'quality_score' in df.columns else float('nan')
                    extra = f" | {diag}" if diag else ""
                    if verbose:
                        logging.info(f"✅ [{cfg_path.stem}] raw_mean={raw_mean:.3f} | max quality_score(norm)={norm_max:.3f} | tract_count={tc}{extra}")
                except Exception:
                    if verbose:
                        logging.info(f"✅ [{cfg_path.stem}] score={score:.3f} | tract_count={tc}")
                optimized_csvs.append((cfg, opt_csv, score, tc))
            else:
                logging.error(f"❌ [{cfg_path.stem}] {status}")
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futs = {ex.submit(run_combo, i, cfg_path, combo_out, verbose_flag): (i, cfg_path) for i, cfg_path, combo_out, verbose_flag in tasks}
            for fut in as_completed(futs):
                i, cfg_path = futs[fut]
                try:
                    cfg, opt_csv, score, tc, status, diag = fut.result()
                except Exception as e:
                    logging.error(f"❌ [{cfg_path.stem}] exception: {e}")
                    continue
                if status == 'ok':
                    try:
                        df = pd.read_csv(opt_csv)
                        raw_mean = float(df['quality_score_raw'].mean()) if 'quality_score_raw' in df.columns else float('nan')
                        norm_max = float(df['quality_score'].max()) if 'quality_score' in df.columns else float('nan')
                        extra = f" | {diag}" if diag else ""
                        if verbose:
                            logging.info(f"✅ [{cfg_path.stem}] raw_mean={raw_mean:.3f} | max quality_score(norm)={norm_max:.3f} | tract_count={tc}{extra}")
                    except Exception:
                        if verbose:
                            logging.info(f"✅ [{cfg_path.stem}] score={score:.3f} | tract_count={tc}")
                    optimized_csvs.append((cfg, opt_csv, score, tc))
                else:
                    logging.error(f"❌ [{cfg_path.stem}] {status}")

    # After running all combos, aggregate diagnostics.json files to a wave-level CSV
    try:
        import csv as _csv
        diag_rows = []
        for child in combos_dir.iterdir():
            if child.is_dir() and child.name.startswith('sweep_'):
                j = child / 'diagnostics.json'
                if j.exists():
                    try:
                        rec = json.loads(j.read_text())
                        row = {
                            'wave': rec.get('wave'),
                            'sweep_id': child.name,
                            'status': rec.get('status'),
                            'combo_index': rec.get('combo_index'),
                            'total_combinations': rec.get('total_combinations'),
                            'sampler': rec.get('sampler'),
                            'thread_count': rec.get('thread_count'),
                            'tract_count': rec.get('tract_count'),
                            'selection_score': rec.get('selection_score'),
                            'quality_score_raw_mean': rec.get('quality_score_raw_mean'),
                            'quality_score_norm_max': rec.get('quality_score_norm_max'),
                            'density_mean': (rec.get('aggregates') or {}).get('density_mean'),
                            'global_efficiency_weighted_mean': (rec.get('aggregates') or {}).get('global_efficiency_weighted_mean'),
                            'small_worldness_binary_mean': (rec.get('aggregates') or {}).get('small_worldness_binary_mean'),
                            'small_worldness_weighted_mean': (rec.get('aggregates') or {}).get('small_worldness_weighted_mean'),
                        }
                        diag_rows.append(row)
                    except Exception:
                        pass
        if diag_rows:
            out_csv = wave_output_dir / 'combo_diagnostics.csv'
            cols = [
                'wave','sweep_id','status','combo_index','total_combinations','sampler','thread_count','tract_count',
                'selection_score','quality_score_raw_mean','quality_score_norm_max','density_mean','global_efficiency_weighted_mean',
                'small_worldness_binary_mean','small_worldness_weighted_mean'
            ]
            with out_csv.open('w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in sorted(diag_rows, key=lambda r: (r.get('combo_index') or 0)):
                    w.writerow({k: r.get(k) for k in cols})
            logging.info(f"📝 Wrote wave-level combo diagnostics: {out_csv}")
    except Exception as e:
        logging.warning(f"⚠️  Could not write wave-level diagnostics CSV: {e}")

    if not optimized_csvs:
        logging.error("❌ No successful combinations completed Step 02")
        return False

    # Summary message for non-verbose mode
    if not verbose:
        success_count = len(optimized_csvs)
        logging.info(f"✅ Completed {success_count}/{len(combos)} parameter combinations successfully")

    # Choose best combination by highest max quality_score
    best = None
    best_score = -1.0
    best_tc = None
    eps = 1e-4
    for cfg_path, opt_csv, sc, tc in optimized_csvs:
        if verbose:
            logging.info(f"📊 {cfg_path.stem}: selection_score={sc:.3f} | tract_count={tc}")
        if (sc > best_score + eps) or (abs(sc - best_score) <= eps and (best_tc is None or (tc != -1 and tc < best_tc))):
            best_score = sc
            best_tc = tc
            best = (cfg_path, opt_csv)

    if not best:
        logging.error("❌ Could not determine best combination (no scores)")
        return False

    # Step 03: run optimal selection for the best combo into wave root
    best_cfg, best_opt_csv = best
    logging.info(f"🏆 Selected best parameters: {best_cfg.name} (selection_score={best_score:.3f}, tract_count={best_tc})")
    step03_dir = wave_output_dir / '03_selection'
    step03_dir.mkdir(exist_ok=True)
    cmd03 = [
        sys.executable, str(root / 'scripts' / 'optimal_selection.py'),
        '-i', str(best_opt_csv), '-o', str(step03_dir)
    ]
    logging.debug(f"🔧 Step03 cmd: {' '.join(cmd03)}")
    rc3 = subprocess.call(cmd03)
    if rc3 != 0:
        logging.error("❌ Step 03 failed for best combination")
        return False

    # Persist selection metadata
    try:
        meta_out = wave_output_dir / 'selected_parameters.json'
        with open(best_cfg, 'r') as _in, meta_out.open('w') as _out:
            data = json.load(_in)
            json.dump({'selected_config': data}, _out, indent=2)
        logging.info(f"📝 Selected parameters saved to {meta_out}")
    except Exception:
        pass

    # Optionally prune non-best combo outputs to save space
    if prune_nonbest:
        try:
            for child in combos_dir.iterdir():
                if child.is_dir() and child.name.startswith('sweep_') and (child / '02_optimization').exists():
                    if child != best_opt_csv.parent.parent:  # parent of 02_optimization is combo_out
                        shutil.rmtree(child, ignore_errors=True)
                        logging.info(f"🧹 Pruned {child.name}")
        except Exception as e:
            logging.warning(f"⚠️  Pruning non-best combos failed: {e}")

    logging.info(f"✅ Wave {wave_name} completed successfully")
    return True

def main():
    """Main cross-validation optimizer."""
    parser = argparse.ArgumentParser(description='Cross-Validation Bootstrap Optimizer')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Perform a dry-run: generate configs and summarize actions without executing the pipeline')
    # Print help when no args provided
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    parser.add_argument('-i', '--data-dir', required=True, help='Data directory')
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--extraction-config', help='Override extraction config used in auto-generated waves')
    parser.add_argument('--wave1-config', help='Wave 1 configuration file')
    parser.add_argument('--wave2-config', help='Wave 2 configuration file')
    parser.add_argument('--single-wave', action='store_true', 
                        help='Run single wave instead of cross-validation (uses all subjects for one comprehensive optimization)')
    parser.add_argument('--subjects', type=int, default=3, help='Subjects per wave (default: 3)')
    parser.add_argument('--max-parallel', type=int, default=1, help='Max combinations to run in parallel per wave (default: 1)')
    parser.add_argument('--prune-nonbest', action='store_true', help='After selection, delete non-best combo outputs to save space')
    parser.add_argument('--no-emoji', action='store_true', help='Disable emoji in console output (Windows-safe)')
    parser.add_argument('--verbose', action='store_true', help='Show DSI Studio command for each run in main sweep log')
    
    args = parser.parse_args()

    configure_stdio(args.no_emoji)
    
    # Initialize logging with file handler under output directory
    # Use <output>/optimize as the base for all optimizer artifacts
    base_output = Path(args.output_dir) / 'optimize'
    setup_logging(str(base_output))
    logging.info("🎯 CROSS-VALIDATION BOOTSTRAP OPTIMIZER")
    logging.info("=" * 50)
    logging.info(f"📂 Input data directory: {args.data_dir}")
    logging.info(f"📁 Output directory: {args.output_dir}")
    
    # Create output directory
    # Create base output directory for optimization
    output_dir = base_output
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"📁 Created output directory: {output_dir}")
    
    # Determine wave configurations
    if args.wave1_config and args.wave2_config:
        logging.info("📋 Using provided wave configuration files")
        wave1_config = args.wave1_config
        wave2_config = args.wave2_config
    elif args.config:
        logging.info("📋 Loading master configuration file")
        # Load master config and extract wave configs
        with open(args.config, 'r') as f:
            master_config = json.load(f)
        wave1_config = master_config.get('wave1_config')
        wave2_config = master_config.get('wave2_config')
        
        # If not specified in master config, generate them
        if not wave1_config or not wave2_config:
            logging.info("📝 Generating wave configurations from master config")
            wave1_config, wave2_config = generate_wave_configs(args.data_dir, output_dir, n_subjects=args.subjects)
    else:
        logging.info("📝 Auto-generating default wave configurations")
        # Generate default wave configurations
        if args.single_wave:
            logging.info("🏄 Single wave mode: comprehensive optimization with all specified subjects")
            wave1_config = generate_single_wave_config(
                args.data_dir, output_dir, n_subjects=args.subjects, extraction_cfg=args.extraction_config
            )
            wave2_config = None
        else:
            wave1_config, wave2_config = generate_wave_configs(
                args.data_dir, output_dir, n_subjects=args.subjects, extraction_cfg=args.extraction_config
            )
    
    logging.info(f"📁 Output directory: {output_dir}")
    if args.single_wave:
        logging.info(f"📄 Single wave config: {wave1_config}")
    else:
        logging.info(f"📄 Wave 1 config: {wave1_config}")
        logging.info(f"📄 Wave 2 config: {wave2_config}")
    
    # Record start time
    start_time = time.time()
    
    # Run Wave 1
    logging.info("\\n" + "🌊" * 20)
    logging.info("🌊 RUNNING OPTIMIZATION WAVE")
    logging.info("🌊" * 20)
    wave1_start = time.time()
    wave1_success = run_wave_pipeline(wave1_config, output_dir, max_parallel=args.max_parallel, prune_nonbest=args.prune_nonbest, verbose=args.verbose)
    wave1_duration = time.time() - wave1_start
    logging.info(f"⏱️  Wave completed in {wave1_duration:.1f} seconds")
    
    # Run Wave 2 if not single wave mode
    wave2_success = True
    wave2_duration = 0.0
    if not args.single_wave and wave2_config:
        logging.info("\n" + "🌊" * 20)
        logging.info("🌊 RUNNING VALIDATION WAVE")
        logging.info("🌊" * 20)
        wave2_start = time.time()
        wave2_success = run_wave_pipeline(wave2_config, output_dir, max_parallel=args.max_parallel, prune_nonbest=args.prune_nonbest, verbose=args.verbose)
        wave2_duration = time.time() - wave2_start
        logging.info(f"⏱️  Wave 2 completed in {wave2_duration:.1f} seconds")
    
    # Final summary
    total_duration = time.time() - start_time
    logging.info("\\n" + "=" * 60)
    
    if wave1_success and wave2_success:
        if args.single_wave:
            logging.info("✅ COMPREHENSIVE OPTIMIZATION COMPLETED SUCCESSFULLY")
            logging.info(f"📊 Results saved in: {output_dir}")
            logging.info(f"⏱️  Total runtime: {total_duration:.1f} seconds")
            # For single wave, copy results directly to optimization_results
            # (No further action required here; results are already saved in output_dir)
        else:
            logging.info("✅ CROSS-VALIDATION COMPLETED SUCCESSFULLY")
            logging.info(f"📊 Results saved in: {output_dir}")
            logging.info(f"⏱️  Total runtime: {total_duration:.1f} seconds")
            logging.info(f"   • Wave 1: {wave1_duration:.1f}s")
            logging.info(f"   • Wave 2: {wave2_duration:.1f}s")
    else:
        logging.error("❌ OPTIMIZATION FAILED")
        logging.error(f"⏱️  Runtime before failure: {total_duration:.1f} seconds")
        if not wave1_success:
            logging.error("   • Optimization wave failed")
        if not wave2_success:
            logging.error("   • Validation wave failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
