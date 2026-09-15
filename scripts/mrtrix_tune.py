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
# 1) run MRtrix tckgen (+ optional tcksift2), `reliability.repeats` times
#    (default 2), varying MRTRIX_RNG_SEED per repeat so repeats are
#    reproducible-but-different tracking runs of the same theta
# 2) run tck2connectome for a chosen atlas, per repeat
# 3) write OptiConn-style `*.connectivity.csv` (with region names), per repeat
# 4) compute network measures from the connectome, per repeat
# 5) aggregate + compute OptiConn QA (uses MetricOptimizer quality_score_raw)
#    across all repeats, plus discriminability/repeatability via
#    scripts.reliability, scored across the repeats' tracked matrices
#
# Output layout intentionally mimics OptiConn Step01 conventions, with one
# extra nesting level for repeats:
#   <out>/01_connectivity/<run>/<theta_id>/rep_<k>/results/<atlas>/*.connectivity.csv

from __future__ import annotations

import argparse
import json
import math
import os
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
from scripts.reliability import rank, score_combo
from scripts.sweep_utils import (
    expand_range,
    grid_product,
    lhs_sampling,
    random_sampling,
)
from scripts.utils.mrtrix import write_opticonn_connectivity_csv

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


DEFAULT_REPEATS = 2


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


def _run(cmd: List[str], *, dry_run: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    if dry_run:
        prefix = ""
        seed = (env or {}).get("MRTRIX_RNG_SEED")
        if seed is not None:
            prefix = f"MRTRIX_RNG_SEED={seed} "
        print("DRY-RUN:", prefix + " ".join(map(str, cmd)))
        return
    subprocess.run(cmd, check=True, env=env)


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


def _quality_score_raw_for_theta(theta_root: Path) -> Tuple[float, Dict[str, float]]:
    """Unchanged (pre-repeats) QA computation: aggregate every
    `*.network_measures.csv` `aggregate_network_measures` finds recursively
    under `theta_root` (now spanning every `rep_<k>/results/<atlas>/`
    directory, not just one) and run it through `MetricOptimizer`, exactly as
    before repeats existed. `quality_score_raw`'s meaning is unchanged; this
    now simply pools across repeats because they all live under `theta_root`.
    """
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


def _compute_qa_for_theta(theta_root: Path, reliability_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Repeats-aware theta scoring.

    `quality_score_raw`/`quality_score_raw_by_metric` keep their pre-existing
    meaning (see `_quality_score_raw_for_theta`) and stay in the returned dict
    for backward-compat reporting -- they no longer drive selection.

    `discriminability`/`repeatability`/`rejected` come from
    `scripts.reliability.score_combo(theta_root, reliability_cfg)` +
    `scripts.reliability.rank`, reduced from one row per (atlas, metric) to a
    single best row exactly the way Task 2's `assemble_ok_result` reduces a
    DSI Studio combo's `score_combo` rows: the best-ranked row's
    discriminability/repeatability when at least one (atlas, metric) pair
    passed the reliability gates, else NaN + the pooled rejection reasons.
    """
    qa_raw, qa_by_metric = _quality_score_raw_for_theta(theta_root)

    rows = score_combo(theta_root, reliability_cfg or {})
    ranked = rank(rows)
    if ranked:
        best_row = ranked[0]
        discr = best_row["discriminability"]
        repeatab = best_row["repeatability"]
        rejected = ""
    else:
        discr = float("nan")
        repeatab = float("nan")
        reasons = sorted({r["rejected"] for r in rows if r["rejected"]})
        rejected = "; ".join(reasons) or "no usable atlas/metric pairs"

    return {
        "quality_score_raw": qa_raw,
        "quality_score_raw_by_metric": qa_by_metric,
        "discriminability": discr,
        "repeatability": repeatab,
        "rejected": rejected,
    }


def select_best_theta(theta_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the winning theta by discriminability, using `scripts.reliability.rank`'s
    ordering (discriminability desc, then repeatability desc, then tract_count asc)
    -- same rule Task 2's `select_best_combo` applies to DSI Studio sweep combos.

    Deviation from strict Task 2 parity, documented: mrtrix_tune's sweep/bayes CLI
    evaluates a single `--subject` per invocation (unlike the DSI Studio wave
    optimizer, which samples several subjects per wave). `scripts.reliability.
    discriminability` compares repeats of the SAME subject against repeats of
    OTHER subjects, so with only one subject ever present, `collect_matrices`
    always finds exactly one subject key and discriminability is structurally
    NaN; `rank()` drops every NaN-discriminability row, so it returns nothing
    for the only workflow this CLI supports. Returning `None` there instead of
    picking a real winner would make sweep/bayes selection silently inert, so
    when `rank()` yields no candidates this falls back to ranking by
    repeatability (a genuine reliability signal, meaningful even for a single
    subject), then `quality_score_raw`, over the thetas that were not
    reliability-gate-rejected outright.
    """
    ranked = rank(theta_results)
    if ranked:
        return ranked[0]

    usable = [r for r in theta_results if not r.get("rejected")]
    if not usable:
        return None

    def _fallback_key(r: Dict[str, Any]) -> Tuple[float, float]:
        repeatab = r.get("repeatability")
        repeatab_key = (
            -repeatab if isinstance(repeatab, (int, float)) and not math.isnan(repeatab) else 0.0
        )
        qa = r.get("quality_score_raw")
        qa_key = -qa if isinstance(qa, (int, float)) and not math.isnan(qa) else 0.0
        return (repeatab_key, qa_key)

    return sorted(usable, key=_fallback_key)[0]


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
    repeats: Optional[int] = None,
) -> Dict[str, Any]:
    theta_dir = out_base / theta_id
    if theta_dir.exists() and any(theta_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing results for {theta_id}: {theta_dir}\n"
            "Pass --overwrite to overwrite, or use a new --run-name/output-dir."
        )

    reliability_cfg = _get_nested(cfg, "reliability", {}) or {}
    if repeats is None:
        repeats = reliability_cfg.get("repeats", DEFAULT_REPEATS)
    repeats = max(1, int(repeats))

    outputs_cfg = ((cfg.get("mrtrix", {}) or {}).get("tck2connectome", {}) or {}).get(
        "outputs"
    )
    output_specs: List[Dict[str, Any]]
    if isinstance(outputs_cfg, list) and outputs_cfg:
        output_specs = [dict(x) for x in outputs_cfg]
    else:
        output_specs = _default_connectome_outputs()

    emitted_metrics: List[str] = []
    rep_dirs: List[Path] = []

    for k in range(1, repeats + 1):
        rep_dir = theta_dir / f"rep_{k}"
        results_dir = rep_dir / "results" / atlas
        if not dry_run:
            results_dir.mkdir(parents=True, exist_ok=True)
        rep_dirs.append(rep_dir)

        tractogram = rep_dir / "tractogram.tck"

        # MRTRIX_RNG_SEED makes tckgen's tracking reproducible for a given
        # value and different across values (confirmed: same seed -> byte
        # identical tracks; different seed -> different tracks; unset ->
        # non-deterministic). Varying it per repeat is this backend's
        # equivalent of Task 2's per-repeat `tracking_parameters.random_seed`.
        env = os.environ.copy()
        env["MRTRIX_RNG_SEED"] = str(k)

        # 1) tckgen
        cmd = _tckgen_cmd(bundle, tractogram, cfg, theta, nthreads=nthreads, enable_act=enable_act)
        if overwrite:
            cmd.append("-force")
        _run(cmd, dry_run=dry_run, env=env)

        # 2) tcksift2 (optional)
        weights_in: Optional[Path] = None
        if enable_sift2:
            weights_in = rep_dir / "sift2_streamlineweights.csv"
            mu_out = rep_dir / "sift2_mu.txt"
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
            _run(cmd, dry_run=dry_run, env=env)

        # 3-5) tck2connectome -> OptiConn CSV -> network measures (for each output metric)
        for spec in output_specs:
            name = str(spec.get("name", "count"))
            weighted = bool(spec.get("weighted", False))
            metric_token = name

            weights_for_this: Optional[Path] = weights_in if weighted else None
            if weighted and weights_in is None:
                # Skip metrics that require weights if SIFT2 not enabled.
                continue

            raw_connectome = rep_dir / f"{subject}_{atlas}.{metric_token}.connectome_raw.csv"
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
            _run(cmd, dry_run=dry_run, env=env)

            if not dry_run:
                write_opticonn_connectivity_csv(
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
            if metric_token not in emitted_metrics:
                emitted_metrics.append(metric_token)

    if dry_run:
        rec = {
            "theta_id": theta_id,
            "params": theta,
            "quality_score_raw": float("nan"),
            "quality_score_raw_by_metric": {},
            "discriminability": float("nan"),
            "repeatability": float("nan"),
            "rejected": "dry-run",
            "emitted_metrics": emitted_metrics,
            "repeats": repeats,
            "theta_dir": str(theta_dir),
            "results_dir": str(theta_dir / "rep_1" / "results" / atlas),
            "dry_run": True,
        }
        return rec

    # 6) Compute QA score (raw) + reliability (discriminability/repeatability)
    qa = _compute_qa_for_theta(theta_dir, reliability_cfg)

    rec = {
        "theta_id": theta_id,
        "params": _to_jsonable(theta),
        "quality_score_raw": qa["quality_score_raw"],
        "quality_score_raw_by_metric": qa["quality_score_raw_by_metric"],
        "discriminability": qa["discriminability"],
        "repeatability": qa["repeatability"],
        "rejected": qa["rejected"],
        "emitted_metrics": emitted_metrics,
        "repeats": repeats,
        "theta_dir": str(theta_dir),
        "results_dir": str(theta_dir / "rep_1" / "results" / atlas),
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

    df = pd.DataFrame(results).sort_values(
        ["discriminability", "repeatability", "quality_score_raw"], ascending=False
    )
    out_csv = out_dir / f"mrtrix_sweep_results_{args.run_name}.csv"
    df.to_csv(out_csv, index=False)

    # Selection is discriminability-first (scripts.reliability.rank order, with
    # the single-subject fallback documented on select_best_theta);
    # quality_score_raw stays a reported field, not the sort key.
    best = select_best_theta(results)
    if best:
        (out_dir / f"mrtrix_best_{args.run_name}.json").write_text(json.dumps(best, indent=2))

    print(f"Wrote sweep table: {out_csv}")
    if best:
        print(
            f"Best theta: {best['theta_id']} (discriminability={best['discriminability']:.4f}, "
            f"repeatability={best['repeatability']:.4f}, quality_score_raw={best['quality_score_raw']:.4f})"
        )
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

        # skopt minimizes, so negate. The Bayesian exploration objective stays
        # quality_score_raw (unchanged) -- only the final selection below is
        # rewired to discriminability; switching skopt's own ask/tell objective
        # to a reliability signal that is frequently NaN for this CLI's
        # single-subject runs (see select_best_theta) is a separate, larger
        # change this task does not make.
        y = -float(rec["quality_score_raw"])
        opt.tell(x, y)

        # Progress
        best_so_far = select_best_theta(results)
        if best_so_far:
            print(
                f"Iter {i}/{args.n_iterations}: rawQA={rec['quality_score_raw']:.4f} "
                f"| best_discr={best_so_far['discriminability']:.4f} (rawQA={best_so_far['quality_score_raw']:.4f})"
            )
        else:
            print(f"Iter {i}/{args.n_iterations}: rawQA={rec['quality_score_raw']:.4f} | best=none yet")

    if args.dry_run:
        print("DRY-RUN: evaluation skipped; no outputs written")
        return 0

    df = pd.DataFrame(results).sort_values(
        ["discriminability", "repeatability", "quality_score_raw"], ascending=False
    )
    out_csv = out_dir / f"mrtrix_bayes_results_{args.run_name}.csv"
    df.to_csv(out_csv, index=False)

    best = select_best_theta(results)
    if best:
        (out_dir / f"mrtrix_best_{args.run_name}.json").write_text(json.dumps(best, indent=2))

        # Write OptiConn-compatible Bayesian results for 'opticonn select'
        opticonn_results = {
            "best_parameters": best.get("params"),
            "best_quality_score": best.get("quality_score_raw"),
            "best_discriminability": best.get("discriminability"),
            "best_repeatability": best.get("repeatability"),
            "target_modality": "qa",  # MRtrix backend currently optimizes for QA
            "n_iterations": int(args.n_iterations),
            "completed_iterations": len(results),
            "all_iterations": results,
            "run_metadata": {
                "backend": "mrtrix",
                "subject": args.subject,
                "run_name": args.run_name,
            },
        }
        (out_dir / "bayesian_optimization_results.json").write_text(
            json.dumps(opticonn_results, indent=2)
        )

    print(f"Wrote Bayesian table: {out_csv}")
    if best:
        print(
            f"Best theta: {best['theta_id']} (discriminability={best['discriminability']:.4f}, "
            f"repeatability={best['repeatability']:.4f}, quality_score_raw={best['quality_score_raw']:.4f})"
        )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    cfg = _load_or_discover_cfg(args)
    bundle, atlas = _build_bundle(cfg, args.atlas)

    # Load theta from optimal config
    with open(args.optimal_config, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        # selected_candidate.json format
        theta = data[0].get("tracking_parameters", {})
    else:
        # bayesian_optimization_results.json format
        theta = data.get("best_parameters", {})

    if not theta:
        print("Error: No parameters found in optimal config.")
        return 1

    out_dir = Path(args.output_dir)
    run_base = out_dir / "01_connectivity" / args.run_name
    run_base.mkdir(parents=True, exist_ok=True)

    print(f"Applying parameters: {theta}")

    evaluate_theta(
        cfg,
        bundle,
        atlas,
        subject=args.subject,
        out_base=run_base,
        theta_id="applied",
        theta=theta,
        nthreads=args.nthreads,
        enable_act=args.enable_act,
        enable_sift2=args.enable_sift2,
        compute_smallworld=args.smallworld,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"Application completed. Results in {run_base}/applied")
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

    pa = sub.add_parser("apply", parents=[common], help="Apply optimal parameters")
    pa.add_argument("--optimal-config", required=True, help="Path to optimal config JSON")
    pa.set_defaults(func=cmd_apply)

    if len(sys.argv) == 1:
        p.print_help()
        return 0

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
