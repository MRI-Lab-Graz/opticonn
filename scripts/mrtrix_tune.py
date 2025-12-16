#!/usr/bin/env python3
"""MRtrix backend tuning (ADD-ON backend; DSI Studio flows remain untouched).

Author: Karl Koschutnig (MRI-Lab Graz)
Contact: karl.koschutnig@uni-graz.at
Date:
"""

# This script provides two minimal workflows:
# - sweep: grid/random sampling over a discrete parameter set
# - bayes: Bayesian optimization (skopt) over numeric ranges / categoricals
#
# For each candidate parameter set (theta):
# 1) run MRtrix tckgen (+ optional tcksift2)
# 2) run tck2connectome for a chosen atlas
# 3) write OptiConn-style `*.connectivity.csv` (with region names)
# 4) compute network measures from the connectome
# 5) aggregate + compute OptiConn QA (uses MetricOptimizer quality_score_raw)
#
# Output layout intentionally mimics OptiConn Step01 conventions:
#   <out>/01_connectivity/<run>/<theta_id>/results/<atlas>/*.connectivity.csv

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Allow running as either:
#   python -m scripts.mrtrix_tune ...
# or
#   python scripts/mrtrix_tune.py ...
if __package__ is None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.aggregate_network_measures import aggregate_network_measures
from scripts.metric_optimizer import MetricOptimizer
from scripts.sweep_utils import (
    expand_range,
    grid_product,
    lhs_sampling,
    random_sampling,
)

try:
    from skopt import Optimizer as SkOptimizer
    from skopt.space import Categorical, Integer, Real

    _SKOPT_OK = True
except Exception:
    _SKOPT_OK = False

# Local helper script functions
from scripts.compute_network_measures_from_connectivity import (
    compute_measures,
    write_network_measures_csv,
)

try:
    # Optional: used when the user provides --derivatives-dir/--qsirecon-dir
    from scripts.mrtrix_discover_bundle import build_config_json, discover_bundle

    _DISCOVERY_OK = True
except Exception:
    _DISCOVERY_OK = False


def _default_connectome_outputs() -> List[Dict[str, Any]]:
    # Minimal default set: count (strength-like) + meanlength (distance-like)
    return [
        {"name": "count", "weighted": False, "scale": None, "stat_edge": None},
        {"name": "meanlength", "weighted": False, "scale": "length", "stat_edge": "mean"},
    ]


@dataclass(frozen=True)
class Bundle:
    wm_fod: Path
    act_5tt_or_hsvs: Optional[Path]
    parcellation_dseg: Path
    parcellation_labels: Path


def _run(cmd: List[str], *, dry_run: bool = False) -> None:
    if dry_run:
        print("DRY-RUN:", " ".join(map(str, cmd)))
        return
    subprocess.run(cmd, check=True)


def _refuse_unsafe_output_dir(output_dir: Path) -> None:
    # Safety: never write outputs under upstream preprocessing folders.
    lowered = {p.lower() for p in output_dir.resolve().parts}
    if "qsiprep" in lowered or "qsirecon" in lowered:
        raise ValueError(
            f"Refusing to write outputs under qsiprep/qsirecon: {output_dir}\n"
            "Use something like <derivatives>/opticonn/... instead."
        )


def _ensure_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _to_jsonable(obj: Any) -> Any:
    # Convert numpy scalar types (e.g., int64/float64/bool_) and nested structures.
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _parse_labels_lookup(path: Path) -> Dict[int, str]:
    # Accept TSV or whitespace-delimited two-column text.
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # allow tab or spaces
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            mapping[idx] = " ".join(parts[1:])
    return mapping


def _load_raw_connectome_matrix(path: Path) -> np.ndarray:
    # tck2connectome outputs whitespace-delimited numeric matrix.
    for delim in (None, ","):
        try:
            mat = np.loadtxt(path, delimiter=delim)
            if mat.ndim == 1:
                mat = np.atleast_2d(mat)
            return mat
        except Exception:
            continue
    raise ValueError(f"Could not parse connectome matrix: {path}")


def _write_opticonn_connectivity_csv(
    raw_matrix_path: Path,
    labels_lookup: Path,
    out_csv: Path,
) -> None:
    mat = _load_raw_connectome_matrix(raw_matrix_path)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Connectome matrix must be square; got {mat.shape}")

    lookup = _parse_labels_lookup(labels_lookup)
    n = mat.shape[0]
    labels = [lookup.get(i, str(i)) for i in range(1, n + 1)]

    df = pd.DataFrame(mat, index=labels, columns=labels)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)


def _select_parcellation(cfg: Dict[str, Any], atlas: str | None) -> Tuple[str, Path, Path]:
    parcs = (
        cfg.get("inputs", {})
        .get("bundle", {})
        .get("parcellations", [])
        or []
    )
    if not parcs:
        raise ValueError("No parcellations configured under inputs.bundle.parcellations")

    if atlas:
        for p in parcs:
            if str(p.get("name")) == atlas:
                return atlas, Path(p["dseg"]), Path(p["labels_tsv"])
        raise ValueError(f"Atlas '{atlas}' not found in config parcellations")

    # default: first entry
    p0 = parcs[0]
    name = str(p0.get("name") or "atlas")
    return name, Path(p0["dseg"]), Path(p0["labels_tsv"])


def _build_bundle(cfg: Dict[str, Any], atlas: str | None) -> Tuple[Bundle, str]:
    bundle_cfg = cfg.get("inputs", {}).get("bundle", {})
    atlas_name, dseg, labels = _select_parcellation(cfg, atlas)

    # Paths can be absolute or relative; we treat as absolute on disk.
    act_val = bundle_cfg.get("act_5tt_or_hsvs")
    return (
        Bundle(
            wm_fod=_ensure_path(bundle_cfg["wm_fod"]),
            act_5tt_or_hsvs=_ensure_path(act_val) if act_val else None,
            parcellation_dseg=_ensure_path(dseg),
            parcellation_labels=_ensure_path(labels),
        ),
        atlas_name,
    )


def _get_nested(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_bool(v: Any) -> bool:
    return bool(v) is True


def _tckgen_cmd(
    bundle: Bundle,
    tractogram_out: Path,
    cfg: Dict[str, Any],
    theta: Dict[str, Any],
    nthreads: int,
    enable_act: bool,
) -> List[str]:
    tcfg = (cfg.get("mrtrix", {}) or {}).get("tckgen", {}) or {}

    def get(name: str, fallback: Any = None) -> Any:
        # theta overrides via keys like tckgen.cutoff
        tkey = f"tckgen.{name}"
        if tkey in theta:
            return theta[tkey]
        if name in tcfg:
            return tcfg[name]
        return fallback

    cmd = ["tckgen", str(bundle.wm_fod), str(tractogram_out)]

    algorithm = get("algorithm", "iFOD2")
    if algorithm:
        cmd += ["-algorithm", str(algorithm)]

    # Seeding
    seed = tcfg.get("seed", {}) or {}
    seed_type = str(seed.get("type", "dynamic"))
    if seed_type == "dynamic":
        cmd += ["-seed_dynamic", str(bundle.wm_fod)]
    elif seed_type == "image":
        img = seed.get("image")
        if not img:
            raise ValueError("tckgen.seed.type=image but no tckgen.seed.image provided")
        cmd += ["-seed_image", str(_ensure_path(img))]
    elif seed_type == "gmwmi":
        img = seed.get("image")
        if not img:
            raise ValueError("tckgen.seed.type=gmwmi requires a GM-WM interface image in tckgen.seed.image")
        cmd += ["-seed_gmwmi", str(_ensure_path(img))]

    # Core tunables
    select = get("select")
    if select is not None:
        cmd += ["-select", str(int(select))]

    for opt in ("cutoff", "angle", "step", "minlength", "maxlength", "power"):
        val = get(opt)
        if val is None:
            continue
        cmd += [f"-{opt}", str(val)]

    # ACT (optional) + flags that require it
    if enable_act:
        if bundle.act_5tt_or_hsvs is None:
            raise ValueError("--enable-act was set but bundle.act_5tt_or_hsvs is missing")
        cmd += ["-act", str(bundle.act_5tt_or_hsvs)]
        if _as_bool(get("backtrack", False)):
            cmd.append("-backtrack")
        if _as_bool(get("crop_at_gmwmi", False)):
            cmd.append("-crop_at_gmwmi")

    cmd += ["-nthreads", str(int(nthreads))]
    return cmd


def _tcksift2_cmd(
    bundle: Bundle,
    tractogram: Path,
    weights_out: Path,
    mu_out: Path,
    cfg: Dict[str, Any],
    nthreads: int,
    enable_act: bool,
) -> List[str]:
    cmd = [
        "tcksift2",
        str(tractogram),
        str(bundle.wm_fod),
        str(weights_out),
        "-out_mu",
        str(mu_out),
        "-nthreads",
        str(int(nthreads)),
    ]
    if enable_act:
        if bundle.act_5tt_or_hsvs is None:
            raise ValueError("--enable-act was set but bundle.act_5tt_or_hsvs is missing")
        cmd += ["-act", str(bundle.act_5tt_or_hsvs)]
    return cmd


def _load_or_discover_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "config", None):
        return _load_json(Path(args.config))

    if not _DISCOVERY_OK:
        raise ImportError(
            "Bundle discovery is not available. Ensure scripts/mrtrix_discover_bundle.py is importable."
        )

    derivatives_dir = getattr(args, "derivatives_dir", None)
    qsirecon_dir = getattr(args, "qsirecon_dir", None)
    qsiprep_dir = getattr(args, "qsiprep_dir", None)
    session = getattr(args, "session", None)
    workflow_hint = getattr(args, "workflow_hint", None)

    if derivatives_dir is None and qsirecon_dir is None:
        raise ValueError("Provide --config or --derivatives-dir or --qsirecon-dir")

    if args.atlas is None:
        raise ValueError("When using discovery mode, --atlas is required")

    found = discover_bundle(
        derivatives_dir=Path(derivatives_dir) if derivatives_dir else None,
        qsirecon_dir=Path(qsirecon_dir) if qsirecon_dir else None,
        qsiprep_dir=Path(qsiprep_dir) if qsiprep_dir else None,
        subject=str(args.subject),
        session=str(session) if session else None,
        atlas=str(args.atlas),
        workflow_hint=str(workflow_hint) if workflow_hint else None,
        allow_missing_act=bool(getattr(args, "allow_missing_act", False)),
    )
    cfg = build_config_json(found, str(args.atlas))

    emit_cfg = getattr(args, "emit_config", None)
    if emit_cfg:
        emit_path = Path(emit_cfg)
        emit_path.parent.mkdir(parents=True, exist_ok=True)
        emit_path.write_text(json.dumps(_to_jsonable(cfg), indent=2))
        print(f"Wrote discovered config: {emit_path}")

    return cfg


def _tck2connectome_cmd(
    tractogram: Path,
    dseg: Path,
    out_raw: Path,
    cfg: Dict[str, Any],
    theta: Dict[str, Any],
    output_spec: Dict[str, Any],
    nthreads: int,
    weights_in: Optional[Path],
) -> List[str]:
    ccfg = (cfg.get("mrtrix", {}) or {}).get("tck2connectome", {}) or {}
    cmd = [
        "tck2connectome",
        str(tractogram),
        str(dseg),
        str(out_raw),
        "-zero_diagonal",
        "-symmetric",
        "-nthreads",
        str(int(nthreads)),
    ]

    # Allow theta override via search_space key.
    ars = theta.get(
        "tck2connectome.assignment_radial_search_mm",
        ccfg.get("assignment_radial_search_mm"),
    )
    if ars is not None:
        cmd += ["-assignment_radial_search", str(ars)]

    scale = output_spec.get("scale")
    if scale == "length":
        cmd.append("-scale_length")

    stat_edge = output_spec.get("stat_edge")
    if stat_edge:
        cmd += ["-stat_edge", str(stat_edge)]

    if weights_in is not None:
        cmd += ["-tck_weights_in", str(weights_in)]

    return cmd


def _compute_qa_for_theta(theta_root: Path) -> Tuple[float, Dict[str, float]]:
    step01 = theta_root
    agg_csv = step01 / "aggregated_network_measures.csv"
    ok = aggregate_network_measures(str(step01), str(agg_csv))
    if not ok:
        raise RuntimeError(f"Failed to aggregate network measures in {step01}")

    df = pd.read_csv(agg_csv)
    optimizer = MetricOptimizer()
    scored = optimizer.compute_quality_scores(df)
    if "quality_score_raw" not in scored.columns:
        raise RuntimeError("quality_score_raw missing from MetricOptimizer output")

    # One row per (atlas, connectivity_metric). Build per-metric map and aggregate.
    by_metric: Dict[str, float] = {}
    for _, row in scored.iterrows():
        metric = str(row.get("connectivity_metric", "unknown"))
        by_metric[metric] = float(row["quality_score_raw"])

    # Overall objective: mean across available metrics to avoid optimizing only `count`.
    vals = list(by_metric.values())
    overall = float(np.mean(vals)) if vals else float("nan")
    return overall, by_metric


def evaluate_theta(
    cfg: Dict[str, Any],
    bundle: Bundle,
    atlas: str,
    subject: str,
    out_base: Path,
    theta_id: str,
    theta: Dict[str, Any],
    nthreads: int,
    enable_act: bool,
    enable_sift2: bool,
    compute_smallworld: bool,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    theta_dir = out_base / theta_id
    if theta_dir.exists() and any(theta_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing results for {theta_id}: {theta_dir}\n"
            "Pass --overwrite to overwrite, or use a new --run-name/output-dir."
        )

    results_dir = theta_dir / "results" / atlas
    if not dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)

    tractogram = theta_dir / "tractogram.tck"

    outputs_cfg = ((cfg.get("mrtrix", {}) or {}).get("tck2connectome", {}) or {}).get(
        "outputs"
    )
    output_specs: List[Dict[str, Any]]
    if isinstance(outputs_cfg, list) and outputs_cfg:
        output_specs = [dict(x) for x in outputs_cfg]
    else:
        output_specs = _default_connectome_outputs()

    # 1) tckgen
    cmd = _tckgen_cmd(bundle, tractogram, cfg, theta, nthreads=nthreads, enable_act=enable_act)
    if overwrite:
        cmd.append("-force")
    _run(cmd, dry_run=dry_run)

    # 2) tcksift2 (optional)
    weights_in: Optional[Path] = None
    if enable_sift2:
        weights_in = theta_dir / "sift2_streamlineweights.csv"
        mu_out = theta_dir / "sift2_mu.txt"
        cmd = _tcksift2_cmd(
            bundle,
            tractogram,
            weights_out=weights_in,
            mu_out=mu_out,
            cfg=cfg,
            nthreads=nthreads,
            enable_act=enable_act,
        )
        if overwrite:
            cmd.append("-force")
        _run(cmd, dry_run=dry_run)

    # 3-5) tck2connectome -> OptiConn CSV -> network measures (for each output metric)
    emitted_metrics: List[str] = []
    for spec in output_specs:
        name = str(spec.get("name", "count"))
        weighted = bool(spec.get("weighted", False))
        metric_token = name

        weights_for_this: Optional[Path] = weights_in if weighted else None
        if weighted and weights_in is None:
            # Skip metrics that require weights if SIFT2 not enabled.
            continue

        raw_connectome = theta_dir / f"{subject}_{atlas}.{metric_token}.connectome_raw.csv"
        connectivity_csv = results_dir / f"{subject}_{atlas}.{metric_token}.connectivity.csv"
        network_measures_csv = results_dir / (
            f"{subject}_{atlas}.{metric_token}.connectivity.network_measures.csv"
        )

        cmd = _tck2connectome_cmd(
            tractogram,
            bundle.parcellation_dseg,
            raw_connectome,
            cfg,
            theta,
            output_spec=spec,
            nthreads=nthreads,
            weights_in=weights_for_this,
        )
        if overwrite:
            cmd.append("-force")
        _run(cmd, dry_run=dry_run)

        if not dry_run:
            _write_opticonn_connectivity_csv(
                raw_connectome, bundle.parcellation_labels, connectivity_csv
            )

            weight_type = "distance" if metric_token == "meanlength" else "strength"
            measures = compute_measures(
                connectivity_csv,
                compute_smallworld=compute_smallworld,
                smallworld_nrand=20,
                seed=42,
                weight_type=weight_type,
            )
            write_network_measures_csv(measures, network_measures_csv)
        emitted_metrics.append(metric_token)

    if dry_run:
        rec = {
            "theta_id": theta_id,
            "params": theta,
            "quality_score_raw": float("nan"),
            "quality_score_raw_by_metric": {},
            "emitted_metrics": emitted_metrics,
            "theta_dir": str(theta_dir),
            "results_dir": str(results_dir),
            "dry_run": True,
        }
        return rec

    # 6) Compute QA score (raw)
    qa_raw, qa_by_metric = _compute_qa_for_theta(theta_dir)

    rec = {
        "theta_id": theta_id,
        "params": _to_jsonable(theta),
        "quality_score_raw": qa_raw,
        "quality_score_raw_by_metric": qa_by_metric,
        "emitted_metrics": emitted_metrics,
        "theta_dir": str(theta_dir),
        "results_dir": str(results_dir),
    }
    (theta_dir / "theta_result.json").write_text(json.dumps(rec, indent=2))
    return rec


def _build_sweep_candidates(cfg: Dict[str, Any], n_samples: int, seed: int) -> List[Dict[str, Any]]:
    ss = cfg.get("search_space", {}) or {}
    params = ss.get("parameters", {}) or {}

    # For sweep: everything becomes discrete.
    param_values: Dict[str, List[Any]] = {}
    for k, v in params.items():
        if isinstance(v, str):
            param_values[k] = expand_range(v)
        elif isinstance(v, (list, tuple)):
            # If [min,max] numeric range provided, sweep endpoints only unless user provides explicit list.
            if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                param_values[k] = [v[0], v[1]]
            else:
                param_values[k] = list(v)
        else:
            param_values[k] = [v]

    sweep_type = str(ss.get("type", "grid")).lower()
    if sweep_type == "grid":
        return grid_product(param_values)
    if sweep_type == "random":
        return random_sampling(param_values, n_samples=n_samples, seed=seed)
    if sweep_type == "lhs":
        return lhs_sampling(param_values, n_samples=n_samples, seed=seed)
    if sweep_type in {"bayesian", "bayes", "bo"}:
        return random_sampling(param_values, n_samples=n_samples, seed=seed)

    # Default
    return grid_product(param_values)


def _build_bayes_space(cfg: Dict[str, Any]):
    if not _SKOPT_OK:
        raise ImportError("scikit-optimize not available (pip install scikit-optimize)")

    ss = cfg.get("search_space", {}) or {}
    params = ss.get("parameters", {}) or {}

    dimensions = []
    names = []
    for k, v in params.items():
        if isinstance(v, str):
            vals = expand_range(v)
            dimensions.append(Categorical(vals, name=k))
            names.append(k)
            continue

        if isinstance(v, (list, tuple)):
            if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                lo, hi = v
                if isinstance(lo, int) and isinstance(hi, int):
                    dimensions.append(Integer(int(lo), int(hi), name=k))
                else:
                    dimensions.append(Real(float(lo), float(hi), name=k))
                names.append(k)
            else:
                dimensions.append(Categorical(list(v), name=k))
                names.append(k)
            continue

        dimensions.append(Categorical([v], name=k))
        names.append(k)

    return dimensions, names


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_sweep(args: argparse.Namespace) -> int:
    cfg = _load_or_discover_cfg(args)
    if str(cfg.get("backend")) != "mrtrix":
        raise ValueError("Config backend must be 'mrtrix'")

    bundle, atlas = _build_bundle(cfg, args.atlas)

    out_dir = Path(args.output_dir)
    _refuse_unsafe_output_dir(out_dir)

    run_base = out_dir / "01_connectivity" / args.run_name
    run_base.mkdir(parents=True, exist_ok=True)

    candidates = _build_sweep_candidates(cfg, n_samples=args.n_samples, seed=args.seed)
    if args.max_evals is not None:
        candidates = candidates[: int(args.max_evals)]

    results: List[Dict[str, Any]] = []
    for i, theta in enumerate(candidates, 1):
        theta_id = f"theta_{i:03d}"
        enable_sift2 = bool(args.enable_sift2) or bool(
            _get_nested(cfg, "mrtrix.tcksift2.enabled", False)
        )
        rec = evaluate_theta(
            cfg,
            bundle,
            atlas,
            subject=args.subject,
            out_base=run_base,
            theta_id=theta_id,
            theta=theta,
            nthreads=args.nthreads,
            enable_act=args.enable_act,
            enable_sift2=enable_sift2,
            compute_smallworld=args.smallworld,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        results.append(rec)

        if args.dry_run:
            break

    if args.dry_run:
        print("DRY-RUN: evaluation skipped; no outputs written")
        return 0

    df = pd.DataFrame(results).sort_values("quality_score_raw", ascending=False)
    out_csv = out_dir / f"mrtrix_sweep_results_{args.run_name}.csv"
    df.to_csv(out_csv, index=False)

    best = results and max(results, key=lambda r: float(r.get("quality_score_raw", -1e9)))
    if best:
        (out_dir / f"mrtrix_best_{args.run_name}.json").write_text(json.dumps(best, indent=2))

    print(f"Wrote sweep table: {out_csv}")
    if best:
        print(f"Best raw QA: {best['quality_score_raw']:.4f} ({best['theta_id']})")
    return 0


def cmd_bayes(args: argparse.Namespace) -> int:
    cfg = _load_or_discover_cfg(args)
    if str(cfg.get("backend")) != "mrtrix":
        raise ValueError("Config backend must be 'mrtrix'")

    bundle, atlas = _build_bundle(cfg, args.atlas)

    out_dir = Path(args.output_dir)
    _refuse_unsafe_output_dir(out_dir)

    run_base = out_dir / "01_connectivity" / args.run_name
    run_base.mkdir(parents=True, exist_ok=True)

    dimensions, names = _build_bayes_space(cfg)
    opt = SkOptimizer(dimensions, random_state=args.seed)

    results: List[Dict[str, Any]] = []

    for i in range(1, int(args.n_iterations) + 1):
        x = opt.ask()
        theta = {name: val for name, val in zip(names, x)}
        theta_id = f"theta_{i:03d}"

        enable_sift2 = bool(args.enable_sift2) or bool(
            _get_nested(cfg, "mrtrix.tcksift2.enabled", False)
        )
        rec = evaluate_theta(
            cfg,
            bundle,
            atlas,
            subject=args.subject,
            out_base=run_base,
            theta_id=theta_id,
            theta=theta,
            nthreads=args.nthreads,
            enable_act=args.enable_act,
            enable_sift2=enable_sift2,
            compute_smallworld=args.smallworld,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        results.append(rec)

        if args.dry_run:
            break

        # skopt minimizes, so negate
        y = -float(rec["quality_score_raw"])
        opt.tell(x, y)

        # Progress
        best_so_far = max(results, key=lambda r: float(r.get("quality_score_raw", -1e9)))
        print(
            f"Iter {i}/{args.n_iterations}: rawQA={rec['quality_score_raw']:.4f} | best={best_so_far['quality_score_raw']:.4f}"
        )

    if args.dry_run:
        print("DRY-RUN: evaluation skipped; no outputs written")
        return 0

    df = pd.DataFrame(results).sort_values("quality_score_raw", ascending=False)
    out_csv = out_dir / f"mrtrix_bayes_results_{args.run_name}.csv"
    df.to_csv(out_csv, index=False)

    best = results and max(results, key=lambda r: float(r.get("quality_score_raw", -1e9)))
    if best:
        (out_dir / f"mrtrix_best_{args.run_name}.json").write_text(json.dumps(best, indent=2))

    print(f"Wrote Bayesian table: {out_csv}")
    if best:
        print(f"Best raw QA: {best['quality_score_raw']:.4f} ({best['theta_id']})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="MRtrix backend tuning (add-on backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=False, help="MRtrix backend config JSON")
    common.add_argument(
        "--derivatives-dir",
        default=None,
        help="BIDS derivatives root (auto-detect qsiprep/qsirecon underneath) [discovery mode]",
    )
    common.add_argument(
        "--qsirecon-dir",
        default=None,
        help="Explicit qsirecon directory [discovery mode]",
    )
    common.add_argument(
        "--qsiprep-dir",
        default=None,
        help="Explicit qsiprep directory (optional) [discovery mode]",
    )
    common.add_argument(
        "--session",
        default=None,
        help="Session label (e.g., ses-3) [discovery mode]",
    )
    common.add_argument(
        "--workflow-hint",
        default=None,
        help="Substring to disambiguate qsirecon workflow directory (e.g., qsirecon-MRtrix3_act-HSVS) [discovery mode]",
    )
    common.add_argument(
        "--allow-missing-act",
        action="store_true",
        help="Allow discovery to proceed without an ACT tissue image (must not use --enable-act)",
    )
    common.add_argument(
        "--emit-config",
        default=None,
        help="If set (discovery mode), write the discovered config JSON to this path for provenance",
    )
    common.add_argument("--output-dir", required=True, help="Output directory")
    common.add_argument("--subject", required=True, help="Subject label (e.g., sub-1293171)")
    common.add_argument(
        "--atlas",
        default=None,
        help="Atlas name from config.inputs.bundle.parcellations (default: first)",
    )
    common.add_argument("--run-name", default="mrtrix", help="Run name under 01_connectivity")
    common.add_argument("--nthreads", type=int, default=8, help="MRtrix thread count")
    common.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files (passes -force to MRtrix and overwrites theta dirs)",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the MRtrix commands that would run, but do not execute or write outputs",
    )
    common.add_argument(
        "--enable-act",
        action="store_true",
        help="Enable ACT by passing -act <act_5tt_or_hsvs> to tckgen/tcksift2",
    )
    common.add_argument(
        "--enable-sift2",
        action="store_true",
        help="Enable tcksift2 and feed weights into tck2connectome",
    )
    common.add_argument(
        "--smallworld",
        action="store_true",
        help="Compute small-worldness(binary) for QA (slower)",
    )

    ps = sub.add_parser("sweep", parents=[common], help="Grid/random sweep")
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="For random sampling: number of samples (ignored for grid)",
    )
    ps.add_argument(
        "--max-evals",
        type=int,
        default=None,
        help="Hard cap on number of thetas evaluated (useful for quick pilots)",
    )
    ps.set_defaults(func=cmd_sweep)

    pb = sub.add_parser("bayes", parents=[common], help="Bayesian optimization (skopt)")
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--n-iterations", type=int, default=20)
    pb.set_defaults(func=cmd_bayes)

    if len(sys.argv) == 1:
        p.print_help()
        return 0

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
