"""Subject-ordering viewer: how far a parameter choice moves your subjects.

Emits one self-contained HTML file per sweep. A reader drags a slider across
candidate specifications and watches subject ranks on a graph measure reshuffle
against DSI Studio's untouched defaults, with the tracking-noise floor -- what
re-running the identical specification already costs -- shown alongside.

This is reporting, not selection: nothing here feeds scripts.reliability.rank().
See docs/superpowers/specs/2026-09-25-ordering-viewer-design.md.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from scripts.compute_network_measures_from_connectivity import measures_from_matrix
from scripts.variance_decomposition import collect_sweep_matrices

MIN_SUBJECTS_FOR_ORDERING = 3


class ReferenceMissing(RuntimeError):
    """No combo in a wave was flagged as the DSI Studio defaults reference."""


def _combo_meta(sweep_optimize_dir: Path) -> dict[str, dict]:
    """{combo_id: diagnostics.json}, keyed exactly as collect_sweep_matrices keys."""
    meta = {}
    for path in sorted(Path(sweep_optimize_dir).glob("*/combos/sweep_*/diagnostics.json")):
        combo_dir = path.parent
        meta[f"{combo_dir.parent.parent.name}/{combo_dir.name}"] = json.loads(path.read_text())
    return meta


def _references(meta: dict[str, dict], waves: list[str]) -> dict[str, str]:
    refs = {}
    for wave in waves:
        flagged = [cid for cid, m in meta.items()
                   if m.get("wave", cid.split("/")[0]) == wave and m.get("reference")]
        if not flagged:
            raise ReferenceMissing(
                f"{wave} has no combo flagged reference. Re-run the sweep with "
                f"sweep_parameters.reference_candidate set to DSI Studio's defaults; "
                f"displacement is meaningless without that origin."
            )
        refs[wave] = sorted(flagged)[0]
    return refs


def _varying(meta: dict[str, dict]) -> list[str]:
    """Parameters that take more than one value across non-reference combos."""
    seen: dict[str, set] = {}
    for m in meta.values():
        if m.get("reference"):
            continue
        for key, value in (m.get("parameters") or {}).items():
            seen.setdefault(key, set()).add(value)
    return sorted(k for k, v in seen.items() if len(v) > 1)


def collect(sweep_optimize_dir: Path) -> dict:
    """Viewer payload for a completed sweep. See the module docstring."""
    sweep_optimize_dir = Path(sweep_optimize_dir)
    grouped = collect_sweep_matrices(sweep_optimize_dir)
    if not grouped:
        raise ValueError(
            f"no */combos/sweep_* directories with connectivity matrices under {sweep_optimize_dir}"
        )
    meta = _combo_meta(sweep_optimize_dir)
    waves = sorted({cid.split("/")[0] for cid in meta})
    references = _references(meta, waves)

    # {wave: [scan key]}, sorted -- the order that anonymised labels stand for.
    scan_keys: dict[str, list[str]] = {}
    for (atlas, metric), combos in grouped.items():
        for cid, scans in combos.items():
            wave = cid.split("/")[0]
            scan_keys.setdefault(wave, sorted(scans))
    for wave, keys in scan_keys.items():
        if len(keys) < MIN_SUBJECTS_FOR_ORDERING:
            raise ValueError(
                f"{wave} has {len(keys)} subject(s); rank correlation needs at least "
                f"{MIN_SUBJECTS_FOR_ORDERING}"
            )
    subjects = {w: [f"S{i:02d}" for i in range(1, len(k) + 1)] for w, k in scan_keys.items()}

    # values[combo_id][pair][measure] -> [rep1 values, rep2 values], one per subject
    values: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
    measure_names: set[str] | None = None
    for (atlas, metric), combos in sorted(grouped.items()):
        pair = f"{atlas}/{metric}"
        for cid, scans in combos.items():
            keys = scan_keys[cid.split("/")[0]]
            if any(len(scans.get(k, [])) < 2 for k in keys):
                logging.warning("ordering viewer: %s lacks 2 repeats for some subject; skipped", cid)
                continue
            per_repeat = [[measures_from_matrix(scans[k][rep]) for k in keys] for rep in (0, 1)]
            names = set(per_repeat[0][0])
            measure_names = names if measure_names is None else measure_names & names
            slot = values.setdefault(cid, {}).setdefault(pair, {})
            for name in names:
                slot[name] = [[tbl[name] for tbl in rep] for rep in per_repeat]

    measures = sorted(measure_names or ())
    pairs = sorted({p for c in values.values() for p in c})

    combos_out = []
    for cid in sorted(values):
        wave = cid.split("/")[0]
        ref_id = references[wave]
        entry = {
            "id": cid,
            "wave": wave,
            "reference": bool(meta[cid].get("reference")),
            "params": meta[cid].get("parameters") or {},
            "ranks": {}, "rho_noise": {}, "rho_vs_reference": {},
        }
        for pair in pairs:
            if pair not in values[cid]:
                continue
            entry["ranks"][pair] = {}
            entry["rho_noise"][pair] = {}
            entry["rho_vs_reference"][pair] = {}
            for name in measures:
                rep1, rep2 = values[cid][pair][name]
                entry["ranks"][pair][name] = [float(r) for r in rankdata(rep1)]
                entry["rho_noise"][pair][name] = _rho(rep1, rep2)
                ref = values.get(ref_id, {}).get(pair, {}).get(name)
                entry["rho_vs_reference"][pair][name] = _rho(rep1, ref[0]) if ref else None
        combos_out.append(entry)

    return {
        "sweep_dir": str(sweep_optimize_dir),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "pairs": pairs,
        "measures": measures,
        "order_params": _varying(meta),
        "waves": waves,
        "subjects": subjects,
        "reference": references,
        "combos": combos_out,
    }


def _rho(a: list[float], b: list[float]) -> float | None:
    """Spearman of two subject orderings; None when either is constant or too short."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < MIN_SUBJECTS_FOR_ORDERING:
        return None
    a, b = a[finite], b[finite]
    if a.std() == 0 or b.std() == 0:
        return None
    value = float(spearmanr(a, b)[0])
    return None if np.isnan(value) else value
