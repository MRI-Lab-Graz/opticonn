# tests/test_report_acceptance.py
"""`opticonn report` reproduces the study repository's committed sweep-level numbers (proof the move
changed no result). Skipped unless the acceptance data is configured; run once per release."""

import csv
import glob
import math
import os
from pathlib import Path

import pytest

from scripts.report import build_report, flat_row

RESULTS = os.environ.get("OPTICONN_ACCEPT_RESULTS")
BATTERY = os.environ.get("OPTICONN_ACCEPT_BATTERY")
pytestmark = pytest.mark.skipif(not (RESULTS and BATTERY), reason="acceptance data not configured")

DATASETS = ["ds000221", "ds005270", "ds005256", "ds005613", "ds005577", "ds004958", "ds005360"]
FRAGILITY = ["seed_rho", "param_rho", "gap", "effective_dimensionality", "reference_over_between_subject",
             "reference_rho", "n_measures", "n_seed_pairs", "n_param_pairs", "n_seed_measures", "n_param_measures",
             "min_n_subjects", "n_ed_rows", "n_ed_subjects", "n_ed_dropped", "n_reference_pairs",
             "n_reference_ordering_pairs"]
LEADER = ["n_candidates", "discriminability_saturated", "cross_wave_rho", "winner_wave1", "winner_wave2",
          "winners_agree", "top_two_separation", "wave_variation", "winner_determined"]


def _rows(name, key="dataset_id"):
    with open(Path(BATTERY) / name, newline="") as handle:
        csv.field_size_limit(10 ** 9)
        out = {}
        for r in csv.DictReader(handle):
            out.setdefault(r[key], r)                     # dataset-level columns repeat on every metric row
        return out


def _same(new, old):
    if old in ("", None):
        return new in ("", None)
    if new in ("", None):
        return False
    if old in ("True", "False"):
        return str(new) == old
    try:
        return abs(float(new) - float(old)) < 1e-9
    except ValueError:
        return str(new) == old


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    out = {}
    for ds in DATASETS:
        optimize = glob.glob(f"{RESULTS}/{ds}/sweep-*/optimize")[0]
        out[ds] = build_report(optimize, out_dir=tmp_path_factory.mktemp(ds), atlas="AAL3", metric="count", workers=16)
    return out


def test_fragility_dimensionality_and_displacement_match_the_committed_table(reports):
    old, compared = _rows("fragility_summary.csv"), 0
    for ds in DATASETS:
        row = flat_row(reports[ds])
        for col in FRAGILITY:
            assert _same(row.get(col), old[ds][col]), (ds, col, row.get(col), old[ds][col])
            compared += old[ds][col] != ""
    assert compared > 7 * 10


def test_the_leader_test_matches_the_committed_table(reports):
    old, compared = _rows("cross_dataset_summary.csv"), 0
    for ds in DATASETS:
        row = flat_row(reports[ds])
        for col in LEADER:
            assert _same(row.get(col), old[ds][col]), (ds, col, row.get(col), old[ds][col])
            compared += old[ds][col] != ""
    assert compared > 7 * 7


def test_the_density_relation_and_the_reference_verdict_agree_with_the_committed_defaults_table(reports):
    old = _rows("defaults_gates.csv")
    for ds in DATASETS:
        row = flat_row(reports[ds])
        assert str(row["reference_rejected"]) == old[ds]["reference_rejected"], ds
        # the committed numbers average OptiConn's per-combo diagnostics; the report averages repeat tables
        assert abs(float(row["reference_density_percentile"]) - float(old[ds]["reference_density_percentile"])) < 0.02, ds
        a, b = float(row["small_worldness(binary)_vs_density"]), float(old[ds]["small_worldness_vs_density"])
        assert abs(a - b) < 0.01 and a * b > 0, (ds, a, b)
