"""Variance decomposition across a completed OptiConn sweep.

Reports how much the connectome moves under four sources of variation --
tracking noise, parameter choice, between-session, and between-subject -- as
comparable dissimilarity distributions, so a user can judge whether their
parameter choice matters relative to the effect they study.

This is diagnostic reporting, not a selection criterion: nothing here is
consumed by scripts.reliability.rank() or rank_with_fallback(). See
docs/superpowers/specs/2026-09-18-discriminability-noise-floor-design.md
for why discriminability alone saturates and what this adds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.reliability import collect_matrices, edge_vector, _distance
from scripts.utils.discovery import parse_subject_session

MIN_PAIRS_FOR_CONFIDENCE = 10


def collect_sweep_matrices(
    sweep_optimize_dir: Path,
) -> dict[tuple[str, str], dict[str, dict[str, list]]]:
    """{(atlas, metric): {combo_id: {subj_sess_key: [matrix, ...]}}}

    combo_id is "<wave_dir_name>/<sweep_dir_name>", e.g. "wave1/sweep_0001" --
    scoped to the wave so two waves that happen to both have a "sweep_0001"
    (independent grid sampling per wave) are never merged into one candidate.
    """
    result: dict[tuple[str, str], dict[str, dict[str, list]]] = {}
    combo_dirs = sorted(Path(sweep_optimize_dir).glob("*/combos/sweep_*"))
    for combo_dir in combo_dirs:
        combo_id = f"{combo_dir.parent.parent.name}/{combo_dir.name}"
        found = collect_matrices(combo_dir)
        for (atlas, metric), subj_map in found.items():
            result.setdefault((atlas, metric), {})[combo_id] = subj_map
    return result


def _entry(dissimilarities: list[float], reason_if_empty: str) -> dict:
    return {
        "dissimilarities": dissimilarities,
        "available": len(dissimilarities) > 0,
        "reason": None if dissimilarities else reason_if_empty,
    }


def compute_strata(combo_matrices: dict[str, dict[str, list]]) -> dict[str, dict]:
    """Four dissimilarity-pair strata for one (atlas, metric)'s sweep matrices.

    combo_matrices: {combo_id: {subj_sess_key: [matrix, ...]}}, as produced by
    one value of collect_sweep_matrices()'s return dict.
    """
    combo_ids = sorted(combo_matrices)

    vecs_all_reps: dict[tuple[str, str], list] = {}
    vec_rep0: dict[tuple[str, str], object] = {}
    subject_of: dict[str, str | None] = {}
    session_of: dict[str, str | None] = {}

    for combo_id in combo_ids:
        for key, matrices in combo_matrices[combo_id].items():
            vecs = [edge_vector(m) for m in matrices]
            vecs_all_reps[(combo_id, key)] = vecs
            vec_rep0[(combo_id, key)] = vecs[0]
            if key not in subject_of:
                subject, session = parse_subject_session(key)
                subject_of[key] = subject
                session_of[key] = session

    tracking_noise: list[float] = []
    for vecs in vecs_all_reps.values():
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                tracking_noise.append(_distance(vecs[i], vecs[j]))

    key_to_combos: dict[str, list[str]] = {}
    for combo_id, key in vec_rep0:
        key_to_combos.setdefault(key, []).append(combo_id)

    parameter: list[float] = []
    for key, combos in key_to_combos.items():
        combos = sorted(combos)
        for i in range(len(combos)):
            for j in range(i + 1, len(combos)):
                a = vec_rep0[(combos[i], key)]
                b = vec_rep0[(combos[j], key)]
                parameter.append(_distance(a, b))

    between_session: list[float] = []
    between_subject: list[float] = []
    for combo_id in combo_ids:
        keys = sorted(combo_matrices[combo_id])

        subject_to_keys: dict[str, list[str]] = {}
        for key in keys:
            subject = subject_of.get(key)
            session = session_of.get(key)
            if subject and session:
                subject_to_keys.setdefault(subject, []).append(key)
        for subject, sess_keys in subject_to_keys.items():
            sess_keys = sorted(sess_keys)
            for i in range(len(sess_keys)):
                for j in range(i + 1, len(sess_keys)):
                    a = vec_rep0[(combo_id, sess_keys[i])]
                    b = vec_rep0[(combo_id, sess_keys[j])]
                    between_session.append(_distance(a, b))

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                subject_i = subject_of.get(keys[i]) or keys[i]
                subject_j = subject_of.get(keys[j]) or keys[j]
                if subject_i == subject_j:
                    continue
                a = vec_rep0[(combo_id, keys[i])]
                b = vec_rep0[(combo_id, keys[j])]
                between_subject.append(_distance(a, b))

    return {
        "tracking_noise": _entry(
            tracking_noise, "no combo had >=2 tracking repeats for any subject"
        ),
        "parameter": _entry(
            parameter,
            "no subject/session was evaluated under >=2 candidate parameter sets",
        ),
        "between_session": _entry(
            between_session,
            "not available: no subject had >=2 parsed sessions under one candidate "
            "(single-session cohort, or subject/session identifiers did not parse)",
        ),
        "between_subject": _entry(between_subject, "fewer than 2 distinct subjects found"),
    }


def summarize_stratum(entry: dict) -> dict:
    if not entry["available"]:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "iqr_low": None,
            "iqr_high": None,
            "available": False,
            "low_confidence": False,
            "reason": entry["reason"],
        }
    values = np.asarray(entry["dissimilarities"], dtype=float)
    n = len(values)
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "iqr_low": float(np.percentile(values, 25)),
        "iqr_high": float(np.percentile(values, 75)),
        "available": True,
        "low_confidence": n < MIN_PAIRS_FOR_CONFIDENCE,
        "reason": None,
    }


_RATIO_PAIRS = [
    ("parameter_over_between_session", "parameter", "between_session"),
    ("tracking_noise_over_parameter", "tracking_noise", "parameter"),
    ("tracking_noise_over_between_subject", "tracking_noise", "between_subject"),
    ("between_session_over_between_subject", "between_session", "between_subject"),
    ("parameter_over_between_subject", "parameter", "between_subject"),
]


def compute_ratios(summaries: dict[str, dict]) -> dict[str, float | None]:
    ratios: dict[str, float | None] = {}
    for ratio_name, numerator_key, denominator_key in _RATIO_PAIRS:
        numerator = summaries.get(numerator_key)
        denominator = summaries.get(denominator_key)
        if (
            not numerator
            or not denominator
            or not numerator["available"]
            or not denominator["available"]
            or denominator["mean"] == 0
        ):
            ratios[ratio_name] = None
        else:
            ratios[ratio_name] = numerator["mean"] / denominator["mean"]
    return ratios
