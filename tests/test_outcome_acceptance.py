# tests/test_outcome_acceptance.py
"""The new commands reproduce the study repository's committed numbers (proof the move changed nothing)."""

import csv
import os
from pathlib import Path

import pytest

from scripts import measures_table, outcome_curve, pca_curve

SWEEP = os.environ.get("OPTICONN_ACCEPT_SWEEP")
META = os.environ.get("OPTICONN_ACCEPT_METADATA")
BATTERY = os.environ.get("OPTICONN_ACCEPT_BATTERY")
pytestmark = pytest.mark.skipif(not (SWEEP and META and BATTERY), reason="acceptance data not configured")

PRIMARY = "global_efficiency(weighted)"
SEVEN = ["density", "global_efficiency(binary)", "global_efficiency(weighted)", "clustering_coeff_average(binary)",
         "clustering_coeff_average(weighted)", "modularity", "small_worldness(binary)"]


def _csv(path):
    with open(path, newline="") as handle:
        return {r["combo_id"]: r for r in csv.DictReader(handle)}


def _blank(v):
    return v in ("", None)


def _close(a, b):
    if _blank(a) or _blank(b):
        return _blank(a) and _blank(b)
    return abs(float(a) - float(b)) < 1e-9


@pytest.fixture(scope="module")
def measures(tmp_path_factory):
    out = tmp_path_factory.mktemp("accept")
    measures_table.write_measures(SWEEP, out_dir=out / "measures", extended=True, workers=16)
    return out


def test_age_and_sex_curves_match_speccurve(measures):
    variables = str(Path(META) / "participants.tsv")
    for predictor, covariate, prefix in (("age", "sex", ""), ("sex", "age", "sex_")):
        o = measures / predictor
        outcome_curve.main(["-i", SWEEP, "--measures-dir", str(measures / "measures"), "--variables", variables,
                            "--predictor", predictor, "--covariate", covariate, "--measure", PRIMARY, "-o", str(o)])
        new, old = _csv(o / "outcome.csv"), _csv(Path(BATTERY) / "speccurve_ds005270.csv")
        assert set(new) == set(old), predictor
        compared = 0
        for cid, row in new.items():
            assert bool(row["gate"]) == bool(old[cid]["gate"]), (predictor, cid, "gate")
            for stat in ("rho", "p", "ci_low", "ci_high"):
                assert _close(row[f"{stat}:{PRIMARY}"], old[cid][f"{prefix}{stat}"]), (predictor, cid, stat)
            if not _blank(row[f"rho:{PRIMARY}"]) and not _blank(old[cid][f"{prefix}rho"]):
                compared += 1
        committed = sum(1 for r in old.values() if not _blank(r[f"{prefix}rho"]))
        assert compared == committed > 50, (predictor, compared, committed)


def test_the_pca_matches_the_seven_and_twelve_measure_runs(measures):
    variables = str(Path(META) / "participants.tsv")
    for names, committed in ((SEVEN, "pcacurve_ds005270.csv"), (None, "pcacurve12_ds005270.csv")):
        o = measures / committed
        argv = ["-i", SWEEP, "--measures-dir", str(measures / "measures"), "--variables", variables,
                "--predictor", "age", "--covariate", "sex", "-o", str(o)]
        pca_curve.main(argv + (["--measures", *names] if names else []))
        new, old = _csv(o / "pca.csv"), _csv(Path(BATTERY) / committed)
        assert set(new) == set(old), committed
        compared = 0
        for cid, row in new.items():
            if row["gate"]:
                continue
            for key in ("evr1", "evr2", "rho1", "p1", "rho2", "p2"):
                assert _close(row[key], old[cid][key]), (committed, cid, key)
            if not _blank(row["rho1"]) and not _blank(old[cid]["rho1"]):
                compared += 1
        committed_rho1 = sum(1 for cid, r in old.items() if not _blank(r["rho1"]) and not new[cid]["gate"])
        assert compared == committed_rho1 > 50, (committed, compared, committed_rho1)
