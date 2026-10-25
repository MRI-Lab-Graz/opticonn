"""Sweep-level sensitivity statistics as pure functions (no I/O); `opticonn report` assembles them.

Design: docs/superpowers/specs/2026-10-25-sensitivity-report-design.md
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import spearmanr

from scripts.reliability import _distance
from scripts.utils.discovery import parse_subject_session

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


def _names(per: dict) -> list[str]:
    first = next(iter(next(iter(per.values())).values()))
    return sorted(first[0])


def fragility(per: dict) -> dict:
    """Subject-ordering agreement across repeats of one specification (seed) versus across different
    specifications of one wave (param), both on single raw repeats so the arms carry equal noise."""
    nan = float("nan")
    empty = {"seed_rho": nan, "param_rho": nan, "gap": nan, "n_measures": 0, "n_seed_pairs": 0,
             "n_param_pairs": 0, "n_seed_measures": 0, "n_param_measures": 0, "min_n_subjects": nan}
    if not per:
        return empty
    names = _names(per)
    seed_rhos, param_rhos, counts = [], [], []
    seed_measures, param_measures = set(), set()
    for name in names:
        for subjects in per.values():
            keys = sorted(subjects)
            if len(keys) < MIN_SUBJECTS:
                continue
            rho = spearmanr([subjects[k][0][name] for k in keys], [subjects[k][1][name] for k in keys])[0]
            if not np.isnan(rho):
                seed_rhos.append(float(rho))
                seed_measures.add(name)
                counts.append(len(keys))
        rep0 = {c: {k: v[0][name] for k, v in subjects.items()} for c, subjects in per.items()}
        for a, b in itertools.combinations(sorted(rep0), 2):
            if wave_of(a) != wave_of(b):                  # pair only within a wave
                continue
            keys = sorted(set(rep0[a]) & set(rep0[b]))
            if len(keys) < MIN_SUBJECTS:
                continue
            rho = spearmanr([rep0[a][k] for k in keys], [rep0[b][k] for k in keys])[0]
            if not np.isnan(rho):
                param_rhos.append(float(rho))
                param_measures.add(name)
                counts.append(len(keys))
    seed = float(np.mean(seed_rhos)) if seed_rhos else nan
    param = float(np.mean(param_rhos)) if param_rhos else nan
    return {"seed_rho": seed, "param_rho": param, "gap": seed - param, "n_measures": len(names),
            "n_seed_pairs": len(seed_rhos), "n_param_pairs": len(param_rhos),
            "n_seed_measures": len(seed_measures), "n_param_measures": len(param_measures),
            "min_n_subjects": float(min(counts)) if counts else nan}


def effective_dimensionality(per: dict) -> dict:
    """(sum of eigenvalues)^2 / sum of squared eigenvalues of the z-scored measure battery, one row per
    (specification, scan) using the mean of the two repeats; non-finite rows are dropped and counted."""
    nan = float("nan")
    undefined = {"effective_dimensionality": nan, "n_ed_rows": 0, "n_ed_subjects": 0, "n_ed_dropped": 0}
    if not per:
        return undefined
    names = _names(per)
    rows = [[float(np.mean([reps[0][n], reps[1][n]])) for n in names]
            for subjects in per.values() for reps in subjects.values()]
    n_subjects = len({parse_subject_session(s)[0] or s for subjects in per.values() for s in subjects})
    scorable = [r for r in rows if np.all(np.isfinite(r))]
    composition = {"n_ed_rows": len(scorable), "n_ed_subjects": n_subjects, "n_ed_dropped": len(rows) - len(scorable)}
    if n_subjects < MIN_SUBJECTS or len(scorable) < 3:
        return {"effective_dimensionality": nan, **composition}
    x = np.asarray(scorable, dtype=float)
    std = x.std(0, ddof=0)
    std[std == 0] = 1.0
    z = (x - x.mean(0)) / std
    eig = np.linalg.eigvalsh(np.cov(z, rowvar=False))
    eig = eig[eig > 0]
    if eig.size == 0:
        return {"effective_dimensionality": nan, **composition}
    return {"effective_dimensionality": float(eig.sum() ** 2 / (eig ** 2).sum()), **composition}


def _reference_by_wave(reference_ids) -> dict:
    return {wave_of(c): c for c in reference_ids}


def reference_rho(rep1: dict, reference_ids: set) -> dict:
    """Mean Spearman agreement of subject ordering on each measure between every non-reference
    specification and its wave's reference (repeat 1, at least MIN_SUBJECTS shared scans)."""
    nan = float("nan")
    by_wave = _reference_by_wave(reference_ids)
    rhos = []
    if rep1:
        first = next(iter(next(iter(rep1.values())).values()))
        for name in sorted(first):
            vals = {c: {k: v[name] for k, v in subjects.items()} for c, subjects in rep1.items()}
            for c in vals:
                ref = by_wave.get(wave_of(c))
                if c in reference_ids or ref is None or ref not in vals:
                    continue
                keys = sorted(set(vals[c]) & set(vals[ref]))
                if len(keys) < MIN_SUBJECTS:
                    continue
                rho = spearmanr([vals[c][k] for k in keys], [vals[ref][k] for k in keys])[0]
                if not np.isnan(rho):
                    rhos.append(float(rho))
    if not rhos:
        return {"reference_rho": nan, "n_reference_ordering_pairs": 0,
                "note": "no reference specification in the same wave as any other specification"}
    return {"reference_rho": float(np.mean(rhos)), "n_reference_ordering_pairs": len(rhos), "note": None}


def reference_displacement(vectors: dict, reference_ids: set) -> dict:
    """Mean connectome distance (1 - r of the edge vectors) from each non-reference specification to its
    wave's reference, divided by the mean between-subject distance over all specifications."""
    by_wave = _reference_by_wave(reference_ids)
    ref_distances = []
    for c, scans in vectors.items():
        ref = by_wave.get(wave_of(c))
        if c in reference_ids or ref is None or ref not in vectors:
            continue
        for key, vec in scans.items():
            if key in vectors[ref]:
                ref_distances.append(_distance(vec, vectors[ref][key]))
    if not ref_distances:
        return {"available": False, "reference_over_between_subject": None, "n_reference_pairs": 0,
                "note": "no reference specification in the same wave as any other specification"}
    between = []
    for scans in vectors.values():
        keys = sorted(k for k in scans if parse_subject_session(k)[0])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if parse_subject_session(keys[i])[0] != parse_subject_session(keys[j])[0]:
                    between.append(_distance(scans[keys[i]], scans[keys[j]]))
    if not between or np.mean(between) == 0:
        return {"available": False, "reference_over_between_subject": None,
                "n_reference_pairs": len(ref_distances), "note": "between-subject baseline unavailable or zero"}
    return {"available": True, "reference_over_between_subject": float(np.mean(ref_distances) / np.mean(between)),
            "n_reference_pairs": len(ref_distances), "note": None}
