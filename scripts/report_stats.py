"""Sweep-level sensitivity statistics as pure functions (no I/O); `opticonn report` assembles them.

Design: docs/superpowers/specs/2026-10-25-sensitivity-report-design.md
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

MIN_SUBJECTS = 3


def wave_of(combo_id: str) -> str:
    """The wave part of a wave-scoped specification id; empty for a plain id."""
    return combo_id.split("/", 1)[0] if "/" in combo_id else ""


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def leader_test(m1: dict, m2: dict, discr1: list) -> dict:
    """Is the leading specification determined? Margins of the two waves, reference excluded."""
    out = {k: None for k in ("n_candidates", "discriminability_saturated", "cross_wave_rho", "winner_wave1",
                             "winner_wave2", "winners_agree", "top_two_separation", "wave_variation",
                             "winner_determined")}
    notes = []
    shared_all = sorted(set(m1) & set(m2))
    shared = [s for s in shared_all if not _missing(m1[s]) and not _missing(m2[s])]
    out["n_candidates"] = len(shared)
    if len(shared_all) > len(shared):
        notes.append(f"{len(shared_all) - len(shared)} candidate(s) excluded: discriminability_margin blank or NaN")
    if discr1:
        out["discriminability_saturated"] = all(d == 1.0 for d in discr1)
    else:
        notes.append("discriminability_saturated unavailable: no gradeable discriminability in wave 1")
    if len(shared) < 3:
        notes.append(f"fewer than 3 shared candidates ({len(shared)})")
    else:
        a, b = [m1[s] for s in shared], [m2[s] for s in shared]
        if len(set(a)) == 1 or len(set(b)) == 1:
            notes.append("cross_wave_rho unavailable: the candidate ordering was constant across candidates")
        else:
            out["cross_wave_rho"] = float(spearmanr(a, b)[0])
        out["winner_wave1"] = max(shared, key=lambda s: m1[s])
        out["winner_wave2"] = max(shared, key=lambda s: m2[s])
        out["winners_agree"] = out["winner_wave1"] == out["winner_wave2"]
        mean = {s: (m1[s] + m2[s]) / 2 for s in shared}
        top = sorted(mean, key=lambda s: -mean[s])[:2]
        out["top_two_separation"] = abs(mean[top[0]] - mean[top[1]])
        out["wave_variation"] = sum(abs(m1[s] - m2[s]) for s in shared) / len(shared)
        out["winner_determined"] = bool(out["top_two_separation"] > out["wave_variation"])
    out["notes"] = notes
    return out


def density_relation(means: dict, reference_ids: set) -> dict:
    """Spearman of each measure's specification mean against density, over the non-reference specifications."""
    kept = sorted(s for s in means if s not in reference_ids)
    dens = np.array([means[s]["density"] for s in kept], dtype=float)
    out = {}
    for name in sorted({n for s in kept for n in means[s]} - {"density"}):
        vals = np.array([means[s][name] for s in kept], dtype=float)
        ok = np.isfinite(vals) & np.isfinite(dens)
        usable = ok.sum() > 2 and np.ptp(vals[ok]) > 0 and np.ptp(dens[ok]) > 0
        out[f"{name}_vs_density"] = float(spearmanr(vals[ok], dens[ok])[0]) if usable else None
    return out


def reference_percentile(density_by_wave: dict, reference_by_wave: dict) -> float | None:
    """Mean over waves of the share of a wave's other specifications that are less dense than its reference."""
    shares = []
    for wave, dens in density_by_wave.items():
        ref = reference_by_wave.get(wave)
        if ref is None or ref not in dens:
            continue
        others = [d for s, d in dens.items() if s != ref]
        if others:
            shares.append(float(np.mean([d < dens[ref] for d in others])))
    return float(np.mean(shares)) if shares else None
