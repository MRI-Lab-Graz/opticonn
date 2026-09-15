"""Tests for repeat-run discriminability selection in the grid/Bayesian sweep."""

import math

from scripts.cross_validation_bootstrap_optimizer import select_best_combo


def _combo(name, discriminability, repeatability, quality_score_raw, tract_count=100000, rejected=""):
    return {
        "name": name,
        "discriminability": discriminability,
        "repeatability": repeatability,
        "quality_score_raw": quality_score_raw,
        "tract_count": tract_count,
        "rejected": rejected,
    }


def test_selects_highest_discriminability_even_when_quality_score_raw_is_lower():
    combos = [
        _combo("low_discr_high_quality", discriminability=0.55, repeatability=0.9, quality_score_raw=0.99),
        _combo("high_discr_low_quality", discriminability=0.92, repeatability=0.8, quality_score_raw=0.10),
    ]

    winner = select_best_combo(combos)

    assert winner["name"] == "high_discr_low_quality"


def test_ties_on_discriminability_break_by_repeatability_then_tract_count():
    combos = [
        _combo("tied_low_repeatability", discriminability=0.8, repeatability=0.5, quality_score_raw=0.5, tract_count=200000),
        _combo("tied_high_repeatability", discriminability=0.8, repeatability=0.7, quality_score_raw=0.5, tract_count=200000),
    ]

    winner = select_best_combo(combos)

    assert winner["name"] == "tied_high_repeatability"


def test_combo_with_no_usable_repeats_is_excluded_not_crashing():
    combos = [
        _combo("gated_out", discriminability=float("nan"), repeatability=float("nan"), quality_score_raw=0.99, rejected="fewer than 2 repeats for some subject"),
    ]

    winner = select_best_combo(combos)

    assert winner is None


def test_all_combos_gated_returns_none_not_a_crash():
    combos = [
        _combo("gated_a", discriminability=float("nan"), repeatability=float("nan"), quality_score_raw=0.9, rejected="unequal repeats across subjects"),
        _combo("gated_b", discriminability=math.nan, repeatability=math.nan, quality_score_raw=0.8, rejected="density 0.900 outside [0.02, 0.6]"),
    ]

    assert select_best_combo(combos) is None
