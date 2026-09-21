#!/usr/bin/env python3
"""
Cross-Validation Bootstrap Optimizer
===================================

Runs parameter optimization across two bootstrap waves to find
optimal DSI Studio parameters using cross-validation.

Author: Braingraph Pipeline Team
"""

import re
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
from scripts.reliability import rank_with_fallback, resolve_repeats, score_combo
from scripts.utils.discovery import baseline_scans, find_subject_files, select_scans
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
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    handlers = [console_handler]
    if output_dir:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(
                str(Path(output_dir) / f"cross_validation_{timestamp}.log"),
                encoding="utf-8",
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            handlers.append(file_handler)
        except Exception:
            # Fallback to console-only if cannot create file handler
            pass
    logging.basicConfig(level=logging.INFO, handlers=handlers)


def repo_root() -> Path:
    """Return the repository root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def generate_wave_configs(
    data_dir, output_dir, n_subjects: int = 3,
    extraction_cfg: str | None = None,
    sessions_per_subject: int = 2,
):
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
    if not extraction_cfg or str(extraction_cfg).strip().lower() == "none":
        extraction_cfg = "configs/braingraph_default_config.json"

    wave1_config = {
        "test_config": {
            "name": "bootstrap_qa_wave_1",
            "description": "Bootstrap QA Wave 1 - Quick test validation",
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": int(n_subjects),
            "sessions_per_subject": int(sessions_per_subject),
            "random_seed": 42,
            "file_pattern": "*.fz",
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": extraction_cfg,
        },
        "bootstrap": {"n_iterations": 5, "sample_ratio": 0.8},
    }

    # Wave 2 configuration (different random seed)
    wave2_config = {
        "test_config": {
            "name": "bootstrap_qa_wave_2",
            "description": "Bootstrap QA Wave 2 - Cross-validation",
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": int(n_subjects),
            "sessions_per_subject": int(sessions_per_subject),
            "random_seed": 1337,  # Different seed for different subject sample
            "file_pattern": "*.fz",
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": extraction_cfg,
        },
        "bootstrap": {"n_iterations": 5, "sample_ratio": 0.8},
    }

    # Save configurations
    wave1_path = configs_dir / "wave1_config.json"
    wave2_path = configs_dir / "wave2_config.json"

    with open(wave1_path, "w") as f:
        json.dump(wave1_config, f, indent=2)
    with open(wave2_path, "w") as f:
        json.dump(wave2_config, f, indent=2)

    logging.info(f" Generated wave configurations in {configs_dir}")

    return str(wave1_path), str(wave2_path)


def generate_single_wave_config(
    data_dir, output_dir, n_subjects: int = 5,
    extraction_cfg: str | None = None,
    sessions_per_subject: int = 2,
):
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
    if not extraction_cfg or str(extraction_cfg).strip().lower() == "none":
        extraction_cfg = "configs/braingraph_default_config.json"

    wave_config = {
        "test_config": {
            "name": "comprehensive_optimization",
            "description": "Single comprehensive wave for parameter optimization",
        },
        "data_selection": {
            "source_dir": str(data_dir),
            "selection_method": "random",
            "n_subjects": int(n_subjects),
            "sessions_per_subject": int(sessions_per_subject),
            "random_seed": 42,
            "file_pattern": "*.fz",
        },
        "pipeline_config": {
            "steps_to_run": ["01", "02", "03"],
            "extraction_config": extraction_cfg,
        },
        "bootstrap": {"n_iterations": 5, "sample_ratio": 0.8},
    }

    # Save configuration
    wave_path = configs_dir / "comprehensive_wave.json"

    with open(wave_path, "w") as f:
        json.dump(wave_config, f, indent=2)

    logging.info(f" Generated single comprehensive wave configuration in {configs_dir}")

    return str(wave_path)


def merge_bayes_params_into_config(
    bayes_path: Path, base_cfg_path: Path, output_dir: Path
) -> Path:
    """Merge best parameters from Bayesian results into an extraction config."""

    try:
        with open(bayes_path, "r", encoding="utf-8") as f:
            bayes_data = json.load(f)
        best_params = bayes_data.get("best_parameters", {})
    except Exception as e:
        logging.error(f" Failed to read Bayesian results: {e}")
        return base_cfg_path

    if not best_params:
        logging.error(
            " No best_parameters found in Bayesian results; using base config."
        )
        return base_cfg_path

    try:
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logging.error(f" Failed to read base config: {e}")
        return base_cfg_path

    cfg = dict(cfg)
    cfg.setdefault("tracking_parameters", {})
    cfg.setdefault("connectivity_options", {})

    # Promote key parameters into tracking/connectivity sections
    for key, val in best_params.items():
        if key in {
            "fa_threshold",
            "turning_angle",
            "step_size",
            "min_length",
            "max_length",
            "track_voxel_ratio",
        }:
            cfg["tracking_parameters"][key] = val
        elif key == "tract_count":
            cfg["tract_count"] = val
        elif key == "connectivity_threshold":
            cfg["connectivity_options"]["connectivity_threshold"] = val
        else:
            cfg[key] = val

    output_dir.mkdir(parents=True, exist_ok=True)
    seeded_cfg = output_dir / "extraction_seeded_from_bayes.json"
    try:
        seeded_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        logging.info(
            f" Seeded extraction config from Bayesian best parameters: {seeded_cfg}"
        )
        return seeded_cfg
    except Exception as e:
        logging.error(f" Failed to write seeded config: {e}")
        return base_cfg_path


def _score_key(rec: dict) -> float:
    """Best-effort quality score lookup for Bayesian iteration records."""
    for k in ("quality_score", "qa_score", "best_quality_score", "best_qa_score"):
        v = rec.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def load_bayes_top_k_candidates(bayes_path: Path, k: int) -> list[dict]:
    """Return top-K candidate parameter dicts from a Bayesian results JSON."""
    with open(bayes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    iters = data.get("all_iterations") or []
    candidates = []
    seen = set()

    # Sort by score desc, skip faulty, dedupe by param set
    for rec in sorted(iters, key=_score_key, reverse=True):
        if rec.get("faulty") is True:
            continue
        params = rec.get("params")
        if not isinstance(params, dict) or not params:
            continue
        key = tuple((k, params[k]) for k in sorted(params.keys()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(params)
        if len(candidates) >= max(0, int(k)):
            break

    # Ensure best_parameters is included (if present) even if not in all_iterations
    best = data.get("best_parameters")
    if isinstance(best, dict) and best:
        key = tuple((k, best[k]) for k in sorted(best.keys()))
        if key not in seen:
            candidates.insert(0, best)

    return candidates[: max(0, int(k))] if k else candidates


def apply_unmapped_params(cfg: dict, choice: dict, mapping: dict) -> dict:
    """Apply choice keys not present in sweep mapping into a config."""
    cfg = dict(cfg)
    cfg.setdefault("tracking_parameters", {})
    cfg.setdefault("connectivity_options", {})

    for key, val in (choice or {}).items():
        if key in mapping:
            continue
        if key in {
            "fa_threshold",
            "turning_angle",
            "step_size",
            "min_length",
            "max_length",
            "track_voxel_ratio",
            "otsu_threshold",
            "smoothing",
            "dt_threshold",
            "tip_iteration",
        }:
            cfg["tracking_parameters"][key] = val
        elif key == "tract_count":
            cfg["tract_count"] = val
        elif key == "connectivity_threshold":
            cfg["connectivity_options"]["connectivity_threshold"] = val
        else:
            cfg[key] = val
    return cfg


def select_best_combo(results: list[dict]) -> dict | None:
    """Pick the winning combo by discriminability, using `scripts.reliability.rank`'s
    ordering (discriminability desc, then margin desc, then repeatability desc, then tract_count asc).

    Each dict in `results` is a combo-level result record (see `run_combo`) that
    already carries `discriminability`/`repeatability`/`rejected`/`tract_count`
    from `scripts.reliability.score_combo`. `quality_score_raw` is not used as the
    sort key anymore, but stays on the winning record as a reported field.

    Combos with no usable (unrejected, non-NaN) reliability score are excluded
    from selection rather than crashing the sweep -- mirrors how `reliability.rank`
    drops rejected/NaN rows. Returns None if every combo was excluded.
    """
    return rank_with_fallback(results)


def assemble_ok_result(
    combo_out: Path,
    cfg_path: Path,
    reliability_cfg: dict,
    rep_opt_dfs: list,
    rep_agg_dfs: list,
    partial_failures: list[str],
    first_opt_csv: Path | None,
    tract_count: int,
    thread_count: int,
    sweep_meta: dict,
) -> dict:
    """Build the combo-level "ok" result record from at least one successful repeat.

    Pools `quality_score_raw`/`quality_score_norm_max`/aggregate diagnostics across
    every successful repeat's CSVs, then scores the combo's tracked matrices with
    `scripts.reliability.score_combo`/`rank` for `discriminability`/`repeatability`/
    `rejected`. `partial_failures` carries the same "rep_<k>: <reason>" strings
    `run_combo` already collects for repeats that failed -- reusing that shape,
    not inventing a new one -- so a real tracking flake on e.g. 1 of 2 repeats is
    still visible on the record even though the combo overall still went "ok"
    (reliability's own gating, not this function, is what rejects too-thin combos).
    """
    opt_df = pd.concat(rep_opt_dfs, ignore_index=True)
    raw_mean = (
        float(opt_df["quality_score_raw"].mean())
        if "quality_score_raw" in opt_df.columns
        else float("nan")
    )
    norm_max = (
        float(opt_df["quality_score"].max())
        if "quality_score" in opt_df.columns
        else float("nan")
    )

    dens = geff = sw_b = sw_w = float("nan")
    if rep_agg_dfs:
        diag_df = pd.concat(rep_agg_dfs, ignore_index=True)

        def _col_mean(name: str) -> float:
            # aggregate_network_measures.py's groupby().agg(["mean", ...]) flattens
            # multi-level columns to "<name>_mean" -- match that, not the bare name.
            col = f"{name}_mean"
            return float(diag_df[col].mean()) if col in diag_df.columns else float("nan")

        dens = _col_mean("density")
        geff = _col_mean("global_efficiency(weighted)")
        sw_b = _col_mean("small-worldness(binary)")
        sw_w = _col_mean("small-worldness(weighted)")

    # Reliability scoring across all repeats' matrices for this combo.
    combo_rows = score_combo(combo_out, reliability_cfg)
    best_row = rank_with_fallback(combo_rows)
    if best_row is not None:
        discr = best_row["discriminability"]
        margin = best_row.get("discriminability_margin", float("nan"))
        repeatab = best_row["repeatability"]
        rejected = ""
    else:
        discr = float("nan")
        margin = float("nan")
        repeatab = float("nan")
        reasons = sorted({r["rejected"] for r in combo_rows if r["rejected"]})
        rejected = "; ".join(reasons) or "no usable atlas/metric pairs"

    extra_bits = []
    if not np.isnan(dens):
        extra_bits.append(f"density_mean={dens:.4f}")
    if not np.isnan(geff):
        extra_bits.append(f"geff_w_mean={geff:.4f}")
    diag = " ".join(extra_bits)

    return {
        "status": "ok",
        "cfg_path": cfg_path,
        "combo_out": combo_out,
        "opt_csv": first_opt_csv,
        "tract_count": tract_count,
        "thread_count": thread_count,
        "sweep_meta": sweep_meta,
        "quality_score_raw": raw_mean,
        "quality_score_norm_max": norm_max,
        "discriminability": discr,
        "discriminability_margin": margin,
        "repeatability": repeatab,
        "rejected": rejected,
        "partial_failures": list(partial_failures),
        "diag": diag,
        "_aggregates": {"density": dens, "global_efficiency_weighted": geff,
                         "small_worldness_binary": sw_b, "small_worldness_weighted": sw_w},
    }


def load_wave_config(config_file):
    """Load wave configuration."""
    with open(config_file, "r") as f:
        return json.load(f)


def run_wave_pipeline(
    wave_config_file,
    output_base_dir,
    max_parallel: int = 1,
    verbose: bool = False,
    no_emoji: bool = False,
    candidate_combos: list[dict] | None = None,
    dry_run: bool = False,
):
    """Run pipeline for a single wave."""
    logging.info(f" Running pipeline for {wave_config_file}")

    # Load wave configuration
    wave_config = load_wave_config(wave_config_file)
    wave_name = wave_config["test_config"]["name"]

    logging.info(" Wave configuration loaded:")
    logging.info(f"   • Name: {wave_name}")
    logging.info(f"   • Data source: {wave_config['data_selection']['source_dir']}")
    logging.info(
        f"   • Subset size (n_subjects): {wave_config['data_selection'].get('n_subjects')}"
    )
    logging.info(
        f"   • Bootstrap iterations: {wave_config.get('bootstrap', {}).get('n_iterations')}"
    )

    # Create output directory for this wave
    wave_output_dir = Path(output_base_dir) / wave_name
    wave_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f" Created wave output directory: {wave_output_dir}")

    # List all available files in source and save manifest
    try:
        src_dir = Path(wave_config["data_selection"]["source_dir"])
        patterns = [wave_config["data_selection"].get("file_pattern", "*.fz")]
        files = []
        for pat in patterns:
            files.extend(find_subject_files(src_dir, [pat]))
        # Also include .fib.gz if not already covered
        files.extend(find_subject_files(src_dir, ["*.fib.gz"]))
        # Deduplicate (patterns list and the .fib.gz fallback can overlap;
        # find_subject_files only dedupes within its own single call)
        seen = set()
        uniq = []
        for p in files:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        available_manifest = wave_output_dir / "available_files.txt"
        with available_manifest.open("w") as mf:
            for p in uniq:
                mf.write(str(p) + "\n")
        logging.info(f" Available files listed: {available_manifest} ({len(uniq)})")
    except Exception as e:
        logging.warning(f"  Could not list available files: {e}")

    # Determine selection for this wave
    n_subjects = int(wave_config["data_selection"].get("n_subjects") or 3)
    seed = int(wave_config["data_selection"].get("random_seed") or 42)
    random.seed(seed)
    # Prefer .fz, then .fib.gz
    fz_files = [p for p in uniq if str(p).endswith(".fz")]
    fib_files = [p for p in uniq if str(p).endswith(".fib.gz")]
    # Cross-sectional: one baseline scan per subject. Exclusion comes after, so a
    # subject whose baseline is excluded drops out instead of falling back to a
    # later (possibly post-intervention) session.
    pool = baseline_scans(fz_files + fib_files)
    exclude = {str(x) for x in wave_config["data_selection"].get("exclude_scans") or []}
    if exclude:
        # scan ids look like "sub-043_ses-1" (see scripts/qc_gate.py)
        kept = [
            p for p in pool
            if not any(re.match(re.escape(e) + r"(?![A-Za-z0-9])", p.name) for e in exclude)
        ]
        logging.info(
            " Excluded %d baseline scans listed in exclude_scans (those subjects are dropped)",
            len(pool) - len(kept),
        )
        pool = kept
    if not pool:
        logging.error(" No candidate files found for selection")
        return False
    selected = select_scans(pool, n_subjects, seed)
    logging.info(" Staged %d scans, one per subject (n_subjects=%d)", len(selected), n_subjects)
    # Write selected manifest and build staging dir with symlinks
    selected_manifest = wave_output_dir / "selected_files.txt"
    with selected_manifest.open("w") as sf:
        for p in selected:
            sf.write(str(p) + "\n")
    logging.info(f" Selected files listed: {selected_manifest} ({len(selected)})")
    staging_dir = wave_output_dir / "selected_data"
    staging_dir.mkdir(exist_ok=True)
    for p in selected:
        dest = staging_dir / p.name
        try:
            if os.path.lexists(dest):
                # Same target already linked: silent. Different scan, same name: warn.
                if dest.resolve() != p.resolve():
                    logging.warning(
                        "Two selected scans share the file name %s (%s and %s); only the "
                        "first is staged, so the wave will run on fewer scans than selected.",
                        p.name, dest.resolve(), p,
                    )
            else:
                dest.symlink_to(p.resolve())
        except OSError:
            # Fallback to copy if symlink not permitted
            shutil.copy2(p, dest)
    logging.info(f" Staging data directory: {staging_dir}")

    # Run the pipeline for this wave
    root = repo_root()
    extraction_cfg_rel = wave_config.get("pipeline_config", {}).get(
        "extraction_config", "configs/braingraph_default_config.json"
    )
    extraction_cfg = (
        str((root / extraction_cfg_rel).resolve())
        if not Path(extraction_cfg_rel).is_absolute()
        else extraction_cfg_rel
    )
    logging.info(f" Wave '{wave_name}' using extraction config: {extraction_cfg}")

    # Load base extraction config to determine sweep combinations
    try:
        with open(extraction_cfg, "r") as _f:
            base_cfg = json.load(_f)
    except Exception as e:
        logging.error(f" Failed to load extraction config {extraction_cfg}: {e}")
        return False

    sp = base_cfg.get("sweep_parameters") or {}
    param_values, mapping = build_param_grid_from_config({"sweep_parameters": sp})

    reliability_cfg = base_cfg.get("reliability") or {}
    repeats = resolve_repeats(reliability_cfg)

    if candidate_combos is not None:
        combos = list(candidate_combos)
        method = "candidates"
    else:
        # Determine sampling strategy (default: grid over all values)
        sampling = (sp.get("sampling") or {}) if isinstance(sp, dict) else {}
        method = (sampling.get("method") or "grid").lower()
        n_samples = int(sampling.get("n_samples") or 0)
        seed = int(sampling.get("random_seed") or 42)
        if method == "grid" or not param_values:
            combos = grid_product(param_values) if param_values else [{}]
        elif method == "random":
            if n_samples <= 0:
                n_samples = 24
            combos = sweep_random_sampling(param_values, n_samples, seed)
        else:  # lhs
            if n_samples <= 0:
                n_samples = 24
            combos = lhs_sampling(param_values, n_samples, seed)

    # Prepare sweep directories
    sweep_cfg_dir = wave_output_dir / "configs" / "sweep"
    sweep_cfg_dir.mkdir(parents=True, exist_ok=True)
    combos_dir = wave_output_dir / "combos"
    combos_dir.mkdir(parents=True, exist_ok=True)

    # Helper to echo parameters consistently
    preferred_order = [
        "tract_count",
        "connectivity_threshold",
        "otsu_threshold",
        "fa_threshold",
        "min_length",
        "max_length",
        "track_voxel_ratio",
        "turning_angle",
        "step_size",
        "smoothing",
        "dt_threshold",
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
        return ", ".join(items)

    # Execute Step 01+02 for each combination
    logging.info(
        f" Starting parameter sweep for {wave_name}: {len(combos)} combination(s) [method={method}, max_parallel={max_parallel}]"
    )

    base_threads = int(base_cfg.get("thread_count") or 8)
    adj_threads = max(1, base_threads // max(1, int(max_parallel)))

    # Prepare tasks
    tasks = []
    for i, choice in enumerate(combos, 1):
        # Build derived config with thread_count scaling
        derived = apply_param_choice_to_config(base_cfg, choice, mapping)
        derived = apply_unmapped_params(derived, choice, mapping)
        try:
            import datetime as _dt

            derived["sweep_meta"] = {
                "index": i,
                "choice": choice,
                "sampler": method,
                "total_combinations": len(combos),
                "source_config": extraction_cfg,
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
        except Exception:
            pass
        derived["thread_count"] = adj_threads

        cfg_path = sweep_cfg_dir / f"sweep_{i:04d}.json"
        with cfg_path.open("w") as _out:
            json.dump(derived, _out, indent=2)

        combo_out = combos_dir / f"sweep_{i:04d}"
        combo_out.mkdir(parents=True, exist_ok=True)

        # Echo parameters
        logging.info(
            f" Parameters [{i}/{len(combos)}]: {fmt_choice(choice)} | thread_count={adj_threads}"
        )

        tasks.append((i, cfg_path, combo_out, False))

    def run_repeat(
        rep_dir: Path, rep_cfg_path: Path, env: dict, verbose: bool
    ) -> tuple[Path | None, str]:
        """Run step01+aggregate+step02 for a single repeat. Returns (opt_csv, error)."""
        cmd01 = [
            sys.executable,
            "-u",
            str(root / "scripts" / "run_pipeline.py"),
            "--data-dir",
            str(staging_dir),
            "--step",
            "01",
            "--output",
            str(rep_dir),
            "--extraction-config",
            str(rep_cfg_path),
        ]
        if dry_run:
            logging.info(f" [DRY-RUN] {' '.join(cmd01)}")
        else:
            p1 = subprocess.run(cmd01, capture_output=True, text=True, env=env)
            if verbose and p1.stdout:
                for line in p1.stdout.splitlines():
                    if "DSI Studio command:" in line:
                        logging.info(f"[VERBOSE] {line}")
            if p1.returncode != 0:
                try:
                    (rep_dir / "diagnostics.json").write_text(
                        json.dumps(
                            {
                                "status": "failed",
                                "stage": "step01",
                                "wave": wave_name,
                                "rep_dir": str(rep_dir),
                                "config_path": str(rep_cfg_path),
                                "return_code": p1.returncode,
                                "stdout_tail": p1.stdout[-4000:] if p1.stdout else "",
                                "stderr_tail": p1.stderr[-4000:] if p1.stderr else "",
                            },
                            indent=2,
                        )
                    )
                except Exception:
                    pass
                return None, f"step01_failed: rc={p1.returncode}"

        agg_csv = rep_dir / "01_connectivity" / "aggregated_network_measures.csv"
        cmdAgg = [
            sys.executable,
            str(root / "scripts" / "aggregate_network_measures.py"),
            str(rep_dir / "01_connectivity"),
            str(agg_csv),
        ]
        if dry_run:
            logging.info(f" [DRY-RUN] {' '.join(cmdAgg)}")
        elif not agg_csv.exists():
            pAgg = subprocess.run(cmdAgg, capture_output=True, text=True, env=env)
            if pAgg.returncode != 0 or not agg_csv.exists():
                try:
                    (rep_dir / "diagnostics.json").write_text(
                        json.dumps(
                            {
                                "status": "failed",
                                "stage": "aggregate",
                                "wave": wave_name,
                                "rep_dir": str(rep_dir),
                                "config_path": str(rep_cfg_path),
                                "return_code": pAgg.returncode,
                                "stdout_tail": pAgg.stdout[-4000:] if pAgg.stdout else "",
                                "stderr_tail": pAgg.stderr[-4000:] if pAgg.stderr else "",
                            },
                            indent=2,
                        )
                    )
                except Exception:
                    pass
                return None, f"aggregate_failed: rc={pAgg.returncode}"

        step02_dir = rep_dir / "02_optimization"
        step02_dir.mkdir(exist_ok=True)
        cmd02 = [
            sys.executable,
            str(root / "scripts" / "metric_optimizer.py"),
            "-i",
            str(agg_csv),
            "-o",
            str(step02_dir),
        ]
        opt_csv = step02_dir / "optimized_metrics.csv"
        if dry_run:
            logging.info(f" [DRY-RUN] {' '.join(cmd02)}")
            return None, ""
        p2 = subprocess.run(cmd02, capture_output=True, text=True, env=env)
        if p2.returncode != 0 or not opt_csv.exists():
            try:
                (rep_dir / "diagnostics.json").write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "stage": "step02",
                            "wave": wave_name,
                            "rep_dir": str(rep_dir),
                            "config_path": str(rep_cfg_path),
                            "return_code": p2.returncode,
                            "stdout_tail": p2.stdout[-4000:] if p2.stdout else "",
                            "stderr_tail": p2.stderr[-4000:] if p2.stderr else "",
                        },
                        indent=2,
                    )
                )
            except Exception:
                pass
            return None, f"step02_failed: rc={p2.returncode}"
        return opt_csv, ""

    def run_combo(i: int, cfg_path: Path, combo_out: Path, verbose: bool = False) -> dict:
        """Run `repeats` tracking repeats (varying random_seed) for one combination,
        then score it with `scripts.reliability` across the pooled repeats.

        Returns a combo-level result record. `quality_score_raw`/`quality_score_norm_max`
        stay on the record as reported fields; `discriminability`/`repeatability`/`rejected`
        (from `reliability.score_combo`) are what the caller selects on.
        """
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["PYTHONPATH"] = str(root)

        base_choice_cfg = json.loads(cfg_path.read_text())
        try:
            tract_count = int(
                base_choice_cfg.get("sweep_parameters", {}).get(
                    "tract_count", base_choice_cfg.get("tract_count", -1)
                )
            )
            thread_count = int(base_choice_cfg.get("thread_count") or -1)
            sweep_meta = base_choice_cfg.get("sweep_meta") or {}
        except Exception:
            tract_count, thread_count, sweep_meta = -1, -1, {}

        rep_opt_dfs = []
        rep_agg_dfs = []
        first_opt_csv = None
        errors = []
        for k in range(1, repeats + 1):
            rep_dir = combo_out / f"rep_{k}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            rep_cfg = json.loads(json.dumps(base_choice_cfg))
            rep_cfg.setdefault("tracking_parameters", {})["random_seed"] = k
            rep_cfg_path = rep_dir / "sweep_config.json"
            rep_cfg_path.write_text(json.dumps(rep_cfg, indent=2))

            opt_csv, err = run_repeat(rep_dir, rep_cfg_path, env, verbose)
            if err:
                errors.append(f"rep_{k}: {err}")
                continue
            if dry_run:
                continue
            if first_opt_csv is None:
                first_opt_csv = opt_csv
            try:
                rep_opt_dfs.append(pd.read_csv(opt_csv))
            except Exception as e:
                errors.append(f"rep_{k}: could not read {opt_csv}: {e}")
                continue
            agg_csv = rep_dir / "01_connectivity" / "aggregated_network_measures.csv"
            try:
                rep_agg_dfs.append(pd.read_csv(agg_csv))
            except Exception:
                pass

        if dry_run:
            return {
                "status": "dry_run",
                "cfg_path": cfg_path,
                "combo_out": combo_out,
                "opt_csv": None,
                "tract_count": tract_count,
                "thread_count": thread_count,
                "sweep_meta": sweep_meta,
                "quality_score_raw": float("nan"),
                "quality_score_norm_max": float("nan"),
                "discriminability": float("nan"),
                "discriminability_margin": float("nan"),
                "repeatability": float("nan"),
                "rejected": "dry-run",
                "diag": "",
            }

        if not rep_opt_dfs:
            try:
                (combo_out / "diagnostics.json").write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "stage": "repeats",
                            "wave": wave_name,
                            "combo_dir": str(combo_out),
                            "config_path": str(cfg_path),
                            "errors": errors,
                        },
                        indent=2,
                    )
                )
            except Exception:
                pass
            return {
                "status": "failed",
                "cfg_path": cfg_path,
                "combo_out": combo_out,
                "message": "; ".join(errors) or "all repeats failed",
            }

        result = assemble_ok_result(
            combo_out=combo_out,
            cfg_path=cfg_path,
            reliability_cfg=reliability_cfg,
            rep_opt_dfs=rep_opt_dfs,
            rep_agg_dfs=rep_agg_dfs,
            partial_failures=errors,
            first_opt_csv=first_opt_csv,
            tract_count=tract_count,
            thread_count=thread_count,
            sweep_meta=sweep_meta,
        )
        agg = result.pop("_aggregates")
        raw_mean, norm_max = result["quality_score_raw"], result["quality_score_norm_max"]
        discr, repeatab, rejected = result["discriminability"], result["repeatability"], result["rejected"]
        dens, geff, sw_b, sw_w = agg["density"], agg["global_efficiency_weighted"], agg["small_worldness_binary"], agg["small_worldness_weighted"]

        try:
            diag_json = {
                "status": "ok",
                "wave": wave_name,
                "combo_dir": str(combo_out),
                "config_path": str(cfg_path),
                "combo_index": int(sweep_meta.get("index") or i),
                "total_combinations": int(sweep_meta.get("total_combinations") or -1),
                "sampler": sweep_meta.get("sampler"),
                "parameters": sweep_meta.get("choice"),
                "thread_count": thread_count,
                "tract_count": tract_count,
                "repeats": repeats,
                "quality_score_raw_mean": None if np.isnan(raw_mean) else float(raw_mean),
                "quality_score_norm_max": None if np.isnan(norm_max) else float(norm_max),
                "discriminability": None if np.isnan(discr) else float(discr),
                "discriminability_margin": None
                if np.isnan(result["discriminability_margin"])
                else float(result["discriminability_margin"]),
                "repeatability": None if np.isnan(repeatab) else float(repeatab),
                "rejected": rejected,
                "partial_failures": result["partial_failures"],
                "aggregates": {
                    "density_mean": None if np.isnan(dens) else float(dens),
                    "global_efficiency_weighted_mean": None if np.isnan(geff) else float(geff),
                    "small_worldness_binary_mean": None if np.isnan(sw_b) else float(sw_b),
                    "small_worldness_weighted_mean": None if np.isnan(sw_w) else float(sw_w),
                },
                "files": {
                    "optimized_metrics_csv": str(first_opt_csv) if first_opt_csv else None,
                },
            }
            (combo_out / "diagnostics.json").write_text(json.dumps(diag_json, indent=2))
        except Exception:
            pass

        return result

    results: list[dict] = []
    if max_parallel <= 1:
        for i, cfg_path, combo_out, verbose_flag in tasks:
            res = run_combo(i, cfg_path, combo_out, verbose_flag)
            if res["status"] == "ok":
                extra = f" | {res['diag']}" if res.get("diag") else ""
                if verbose:
                    logging.info(
                        f" [{cfg_path.stem}] quality_score_raw_mean={res['quality_score_raw']:.3f} "
                        f"| discriminability={res['discriminability']:.3f} | tract_count={res['tract_count']}{extra}"
                    )
                results.append(res)
            elif res["status"] == "dry_run":
                results.append(res)
            else:
                logging.error(f" [{cfg_path.stem}] {res.get('message')}")
    else:
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futs = {
                ex.submit(run_combo, i, cfg_path, combo_out, verbose_flag): (i, cfg_path)
                for i, cfg_path, combo_out, verbose_flag in tasks
            }
            for fut in as_completed(futs):
                i, cfg_path = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logging.error(f" [{cfg_path.stem}] exception: {e}")
                    continue
                if res["status"] == "ok":
                    extra = f" | {res['diag']}" if res.get("diag") else ""
                    if verbose:
                        logging.info(
                            f" [{cfg_path.stem}] quality_score_raw_mean={res['quality_score_raw']:.3f} "
                            f"| discriminability={res['discriminability']:.3f} | tract_count={res['tract_count']}{extra}"
                        )
                    results.append(res)
                elif res["status"] == "dry_run":
                    results.append(res)
                else:
                    logging.error(f" [{cfg_path.stem}] {res.get('message')}")

    if dry_run:
        logging.info(
            f" [DRY-RUN] Would run {len(tasks)} combination(s) x {repeats} repeat(s) each "
            f"for wave '{wave_name}'; no DSI Studio or real data required."
        )
        return True

    # After running all combos, aggregate diagnostics.json files to a wave-level CSV
    try:
        import csv as _csv

        diag_rows = []
        for child in combos_dir.iterdir():
            if child.is_dir() and child.name.startswith("sweep_"):
                j = child / "diagnostics.json"
                if j.exists():
                    try:
                        rec = json.loads(j.read_text())
                        row = {
                            "wave": rec.get("wave"),
                            "sweep_id": child.name,
                            "status": rec.get("status"),
                            "combo_index": rec.get("combo_index"),
                            "total_combinations": rec.get("total_combinations"),
                            "sampler": rec.get("sampler"),
                            "thread_count": rec.get("thread_count"),
                            "tract_count": rec.get("tract_count"),
                            "repeats": rec.get("repeats"),
                            "quality_score_raw_mean": rec.get("quality_score_raw_mean"),
                            "quality_score_norm_max": rec.get("quality_score_norm_max"),
                            "discriminability": rec.get("discriminability"),
                            "discriminability_margin": rec.get("discriminability_margin"),
                            "repeatability": rec.get("repeatability"),
                            "rejected": rec.get("rejected"),
                            "density_mean": (rec.get("aggregates") or {}).get(
                                "density_mean"
                            ),
                            "global_efficiency_weighted_mean": (
                                rec.get("aggregates") or {}
                            ).get("global_efficiency_weighted_mean"),
                            "small_worldness_binary_mean": (
                                rec.get("aggregates") or {}
                            ).get("small_worldness_binary_mean"),
                            "small_worldness_weighted_mean": (
                                rec.get("aggregates") or {}
                            ).get("small_worldness_weighted_mean"),
                        }
                        diag_rows.append(row)
                    except Exception:
                        pass
        if diag_rows:
            out_csv = wave_output_dir / "combo_diagnostics.csv"
            cols = [
                "wave",
                "sweep_id",
                "status",
                "combo_index",
                "total_combinations",
                "sampler",
                "thread_count",
                "tract_count",
                "repeats",
                "quality_score_raw_mean",
                "quality_score_norm_max",
                "discriminability",
                "discriminability_margin",
                "repeatability",
                "rejected",
                "density_mean",
                "global_efficiency_weighted_mean",
                "small_worldness_binary_mean",
                "small_worldness_weighted_mean",
            ]
            with out_csv.open("w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in sorted(diag_rows, key=lambda r: (r.get("combo_index") or 0)):
                    w.writerow({k: r.get(k) for k in cols})
            logging.info(f" Wrote wave-level combo diagnostics: {out_csv}")
    except Exception as e:
        logging.warning(f"  Could not write wave-level diagnostics CSV: {e}")

    successful = [r for r in results if r["status"] == "ok"]
    if not successful:
        logging.error(" No successful combinations completed Step 02")
        return False

    # Summary message for non-verbose mode
    if not verbose:
        logging.info(
            f" Completed {len(successful)}/{len(combos)} parameter combinations successfully"
        )

    # Choose best combination by discriminability (scripts.reliability.rank order);
    # quality_score_raw stays a reported field on every record, not the sort key.
    if verbose:
        for res in successful:
            logging.info(
                f" {res['cfg_path'].stem}: discriminability={res['discriminability']} "
                f"| repeatability={res['repeatability']} | quality_score_raw_mean={res['quality_score_raw']:.3f} "
                f"| tract_count={res['tract_count']} | rejected={res['rejected'] or '-'}"
            )

    best = select_best_combo(successful)
    if not best:
        logging.error(
            " Could not determine best combination (no combo passed reliability gates)"
        )
        return False

    best_cfg, best_opt_csv = best["cfg_path"], best["opt_csv"]
    logging.info(
        f" Selected best parameters: {best_cfg.name} (discriminability={best['discriminability']:.3f}, "
        f"repeatability={best['repeatability']:.3f}, quality_score_raw_mean={best['quality_score_raw']:.3f}, "
        f"tract_count={best['tract_count']})"
    )
    step03_dir = wave_output_dir / "03_selection"
    step03_dir.mkdir(exist_ok=True)
    cmd03 = [
        sys.executable,
        str(root / "scripts" / "optimal_selection.py"),
        "-i",
        str(best_opt_csv),
        "-o",
        str(step03_dir),
    ]
    # Propagate --no-emoji flag to subprocess
    if no_emoji:
        cmd03.append("--no-emoji")
    logging.debug(f" Step03 cmd: {' '.join(cmd03)}")
    result03 = subprocess.run(cmd03, check=False)
    if result03.returncode != 0:
        logging.error(" Step 03 failed for best combination")
        return False

    # Persist selection metadata
    try:
        meta_out = wave_output_dir / "selected_parameters.json"
        with open(best_cfg, "r") as _in, meta_out.open("w") as _out:
            data = json.load(_in)
            json.dump({"selected_config": data}, _out, indent=2)
        logging.info(f" Selected parameters saved to {meta_out}")
    except Exception:
        pass

    logging.info(f" Wave {wave_name} completed successfully")
    return True


def main():
    """Main cross-validation optimizer."""
    parser = argparse.ArgumentParser(description="Cross-Validation Bootstrap Optimizer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform a dry-run: generate configs and summarize actions without executing the pipeline",
    )
    # Print help when no args provided
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    parser.add_argument("-i", "--data-dir", required=True, help="Data directory")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument(
        "--extraction-config",
        help="Override extraction config used in auto-generated waves",
    )
    parser.add_argument("--wave1-config", help="Wave 1 configuration file")
    parser.add_argument("--wave2-config", help="Wave 2 configuration file")
    parser.add_argument(
        "--single-wave",
        action="store_true",
        help="Run single wave instead of cross-validation (uses all subjects for one comprehensive optimization)",
    )
    parser.add_argument(
        "--subjects", type=int, default=3, help="Subjects per wave (default: 3)"
    )
    parser.add_argument(
        "--sessions-per-subject",
        type=int,
        default=None,
        help=(
            "Sessions staged per sampled subject (default: 2). With >=2, --subjects counts "
            "SUBJECTS rather than scans, so a wave stages up to subjects x sessions scans "
            "(more tracking compute) but within-subject between-session comparisons become "
            "available to the variance decomposition. Falls back to sampling individual scans "
            "when no subject has enough sessions. Use 1 (or 0) for the legacy scan-level sampling. "
            "Note: discriminability treats each scan as a unit, so a subject's second session "
            "counts as a different subject in the between-subject pool; that is a harder test, "
            "not a test-retest reliability estimate."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max combinations to run in parallel per wave (default: 1)",
    )
    parser.add_argument(
        "--from-bayes",
        help="Path to bayesian_optimization_results.json; seeds best parameters into the extraction config",
    )
    parser.add_argument(
        "--candidates-from-bayes",
        help=(
            "Path to bayesian_optimization_results.json; evaluates top-K Bayesian candidates instead of sampling sweep_parameters"
        ),
    )
    parser.add_argument(
        "--bayes-top-k",
        type=int,
        default=3,
        help="When using --candidates-from-bayes, number of top Bayesian candidates to evaluate (default: 3)",
    )
    parser.add_argument(
        "--random-baseline-k",
        type=int,
        default=0,
        help=(
            "When using --candidates-from-bayes, also evaluate K random candidates drawn from sweep_parameters (default: 0)"
        ),
    )
    parser.add_argument(
        "--candidate-random-seed",
        type=int,
        default=42,
        help="Random seed used for --random-baseline-k (default: 42)",
    )
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Disable emoji in console output (Windows-safe)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show DSI Studio command for each run in main sweep log",
    )

    args = parser.parse_args()

    configure_stdio(args.no_emoji)

    # Initialize logging with file handler under output directory
    # Use <output>/optimize as the base for all optimizer artifacts
    base_output = Path(args.output_dir) / "optimize"
    setup_logging(str(base_output))
    logging.info(" CROSS-VALIDATION BOOTSTRAP OPTIMIZER")
    logging.info("=" * 50)
    logging.info(f" Input data directory: {args.data_dir}")
    logging.info(f" Output directory: {args.output_dir}")

    # Create output directory
    # Create base output directory for optimization
    output_dir = base_output
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f" Created output directory: {output_dir}")

    # Optionally seed extraction config from Bayesian results
    extraction_cfg_path = args.extraction_config
    logging.info(f" Extraction config (arg): {args.extraction_config}")
    logging.info(f" Bayesian seeding (arg): {args.from_bayes}")
    if args.candidates_from_bayes:
        if args.from_bayes:
            logging.info(
                " Both --from-bayes and --candidates-from-bayes provided; ignoring --from-bayes"
            )
        if not args.extraction_config:
            logging.error(
                " --extraction-config is required when using --candidates-from-bayes"
            )
            sys.exit(1)
    elif args.from_bayes:
        extraction_cfg_path = str(
            merge_bayes_params_into_config(
                Path(args.from_bayes),
                Path(args.extraction_config),
                Path(output_dir) / "seeded_from_bayes",
            )
        )

    # Defensive: avoid propagating the literal string "None" into wave configs
    if not extraction_cfg_path or str(extraction_cfg_path).strip().lower() == "none":
        fallback = args.extraction_config or "configs/braingraph_default_config.json"
        logging.warning(
            f" Extraction config resolved to '{extraction_cfg_path}'. Falling back to: {fallback}"
        )
        extraction_cfg_path = fallback

    logging.info(f" Extraction config (resolved): {extraction_cfg_path}")

    candidate_combos = None
    if args.candidates_from_bayes:
        try:
            topk = load_bayes_top_k_candidates(
                Path(args.candidates_from_bayes), args.bayes_top_k
            )
        except Exception as e:
            logging.error(f" Failed to load Bayesian candidates: {e}")
            sys.exit(1)

        # Optional random baseline: sample from extraction config sweep space
        random_baseline = []
        if int(args.random_baseline_k or 0) > 0:
            try:
                with open(extraction_cfg_path, "r", encoding="utf-8") as f:
                    base_cfg = json.load(f)
                sp = base_cfg.get("sweep_parameters") or {}
                param_values, _mapping = build_param_grid_from_config(
                    {"sweep_parameters": sp}
                )
                if not param_values:
                    logging.error(
                        " random baseline requested but sweep_parameters are empty; cannot draw random candidates"
                    )
                    sys.exit(1)
                random_baseline = sweep_random_sampling(
                    param_values,
                    int(args.random_baseline_k),
                    int(args.candidate_random_seed),
                )
            except Exception as e:
                logging.error(f" Failed to generate random baseline candidates: {e}")
                sys.exit(1)

        candidate_combos = list(topk) + list(random_baseline)
        try:
            (output_dir / "candidates.json").write_text(
                json.dumps(
                    {
                        "candidates_from_bayes": str(
                            Path(args.candidates_from_bayes).resolve()
                        ),
                        "bayes_top_k": int(args.bayes_top_k),
                        "random_baseline_k": int(args.random_baseline_k),
                        "candidate_random_seed": int(args.candidate_random_seed),
                        "n_candidates": len(candidate_combos),
                        "candidates": candidate_combos,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logging.info(f" Candidate set written: {output_dir / 'candidates.json'}")
        except Exception as e:
            logging.warning(f" Could not write candidates.json: {e}")

    # Determine wave configurations
    sessions_flag_given = args.sessions_per_subject is not None
    if args.sessions_per_subject is None:
        args.sessions_per_subject = 2
    if args.wave1_config and args.wave2_config:
        if sessions_flag_given:
            logging.warning(
                " --sessions-per-subject is ignored with --wave1-config/--wave2-config; "
                "set data_selection.sessions_per_subject in those files"
            )
        logging.info(" Using provided wave configuration files")
        wave1_config = args.wave1_config
        wave2_config = args.wave2_config
    elif args.config:
        logging.info(" Loading master configuration file")
        # Load master config and extract wave configs
        with open(args.config, "r") as f:
            master_config = json.load(f)
        wave1_config = master_config.get("wave1_config")
        wave2_config = master_config.get("wave2_config")
        if sessions_flag_given and wave1_config and wave2_config:
            logging.warning(
                " --sessions-per-subject is ignored: the master config embeds wave configs; "
                "set data_selection.sessions_per_subject in them"
            )

        # If not specified in master config, generate them
        if not wave1_config or not wave2_config:
            logging.info(" Generating wave configurations from master config")
            wave1_config, wave2_config = generate_wave_configs(
                args.data_dir, output_dir, n_subjects=args.subjects,
                sessions_per_subject=args.sessions_per_subject,
            )
    else:
        logging.info(" Auto-generating default wave configurations")
        # Generate default wave configurations
        if args.single_wave:
            logging.info(
                " Single wave mode: comprehensive optimization with all specified subjects"
            )
            wave1_config = generate_single_wave_config(
                args.data_dir,
                output_dir,
                n_subjects=args.subjects,
                sessions_per_subject=args.sessions_per_subject,
                extraction_cfg=extraction_cfg_path,
            )
            wave2_config = None
        else:
            wave1_config, wave2_config = generate_wave_configs(
                args.data_dir,
                output_dir,
                n_subjects=args.subjects,
                sessions_per_subject=args.sessions_per_subject,
                extraction_cfg=extraction_cfg_path,
            )

    logging.info(f" Output directory: {output_dir}")
    if args.single_wave:
        logging.info(f" Single wave config: {wave1_config}")
    else:
        logging.info(f" Wave 1 config: {wave1_config}")
        logging.info(f" Wave 2 config: {wave2_config}")

    # Record start time
    start_time = time.time()

    # Run Wave 1
    logging.info("\\n" + "" * 20)
    logging.info(" RUNNING OPTIMIZATION WAVE")
    logging.info("" * 20)
    wave1_start = time.time()
    wave1_success = run_wave_pipeline(
        wave1_config,
        output_dir,
        max_parallel=args.max_parallel,
        verbose=args.verbose,
        no_emoji=args.no_emoji,
        candidate_combos=candidate_combos,
        dry_run=args.dry_run,
    )
    wave1_duration = time.time() - wave1_start
    logging.info(f"  Wave completed in {wave1_duration:.1f} seconds")

    # Run Wave 2 if not single wave mode
    wave2_success = True
    wave2_duration = 0.0
    if not args.single_wave and wave2_config:
        logging.info("\n" + "" * 20)
        logging.info(" RUNNING VALIDATION WAVE")
        logging.info("" * 20)
        wave2_start = time.time()
        wave2_success = run_wave_pipeline(
            wave2_config,
            output_dir,
            max_parallel=args.max_parallel,
            verbose=args.verbose,
            no_emoji=args.no_emoji,
            candidate_combos=candidate_combos,
            dry_run=args.dry_run,
        )
        wave2_duration = time.time() - wave2_start
        logging.info(f"  Wave 2 completed in {wave2_duration:.1f} seconds")

    # Final summary
    total_duration = time.time() - start_time
    logging.info("\\n" + "=" * 60)

    if wave1_success and wave2_success:
        if args.single_wave:
            logging.info(" COMPREHENSIVE OPTIMIZATION COMPLETED SUCCESSFULLY")
            logging.info(f" Results saved in: {output_dir}")
            logging.info(f"  Total runtime: {total_duration:.1f} seconds")
            # For single wave, copy results directly to optimization_results
            # (No further action required here; results are already saved in output_dir)
        else:
            logging.info(" CROSS-VALIDATION COMPLETED SUCCESSFULLY")
            logging.info(f" Results saved in: {output_dir}")
            logging.info(f"  Total runtime: {total_duration:.1f} seconds")
            logging.info(f"   • Wave 1: {wave1_duration:.1f}s")
            logging.info(f"   • Wave 2: {wave2_duration:.1f}s")

            # NOTE: indentation here is load-bearing -- this try/except is nested
            # INSIDE the two-wave `else:` body (not a sibling of `if args.single_wave:`),
            # so it is only reachable when wave1_success and wave2_success and
            # NOT args.single_wave. Do not de-indent it back to the `if` block's level.
            try:
                from scripts.variance_decomposition import run as run_variance_decomposition

                run_variance_decomposition(
                    Path(output_dir), Path(output_dir) / "optimization_results"
                )
            except Exception as exc:
                logging.warning("  Variance decomposition skipped: %s", exc)
    else:
        logging.error(" OPTIMIZATION FAILED")
        logging.error(f"  Runtime before failure: {total_duration:.1f} seconds")
        if not wave1_success:
            logging.error("   • Optimization wave failed")
        if not wave2_success:
            logging.error("   • Validation wave failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
