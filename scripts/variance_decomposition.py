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

from scripts.reliability import collect_matrices

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
