import json
import sys

import pandas as pd
import pytest

from scripts.pareto_view import main


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["pareto_view.py", *argv])
    main()


def test_reference_row_excluded_from_front_csv_path(tmp_path, monkeypatch):
    """The reference combo must never be presented as a Pareto-optimal
    candidate. Give it a dominant score/cost so it would sit on the front
    (or dominate everything) if the reference flag were ignored."""
    wave_dir = tmp_path / "wave_1"
    wave_dir.mkdir()
    df = pd.DataFrame(
        [
            {
                "status": "ok",
                "reference": True,
                "tract_count": 1,  # cheapest possible
                "quality_score_raw_mean": 100.0,  # best possible score
                "density_mean": 0.2,
                "atlas": "AAL3",
                "connectivity_metric": "count",
            },
            {
                "status": "ok",
                "reference": False,
                "tract_count": 5000,
                "quality_score_raw_mean": 0.5,
                "density_mean": 0.2,
                "atlas": "AAL3",
                "connectivity_metric": "count",
            },
            {
                "status": "ok",
                "reference": False,
                "tract_count": 8000,
                "quality_score_raw_mean": 0.8,
                "density_mean": 0.2,
                "atlas": "AAL3",
                "connectivity_metric": "count",
            },
        ]
    )
    df.to_csv(wave_dir / "combo_diagnostics.csv", index=False)

    out_dir = tmp_path / "out"
    _run(monkeypatch, [str(wave_dir), "-o", str(out_dir)])

    front = pd.read_csv(out_dir / "pareto_front.csv")
    with_obj = pd.read_csv(out_dir / "pareto_candidates_with_objectives.csv")
    assert not front["reference"].fillna(False).astype(bool).any()
    assert not with_obj["reference"].fillna(False).astype(bool).any()
    assert len(with_obj) == 2


def test_reference_row_excluded_from_front_json_fallback(tmp_path, monkeypatch):
    """Same guarantee via the JSON-fallback row builder (no combo_diagnostics.csv)."""
    wave_dir = tmp_path / "wave_1"
    combos_dir = wave_dir / "combos"
    combos_dir.mkdir(parents=True)

    def _write(name, reference, tract_count, score):
        d = combos_dir / name
        d.mkdir()
        (d / "diagnostics.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "wave": "wave_1",
                    "reference": reference,
                    "tract_count": tract_count,
                    "quality_score_raw_mean": score,
                    "atlas": "AAL3",
                    "connectivity_metric": "count",
                }
            )
        )

    _write("sweep_0001", True, 1, 100.0)
    _write("sweep_0002", False, 5000, 0.5)
    _write("sweep_0003", False, 8000, 0.8)

    out_dir = tmp_path / "out"
    _run(monkeypatch, [str(wave_dir), "-o", str(out_dir)])

    front = pd.read_csv(out_dir / "pareto_front.csv")
    with_obj = pd.read_csv(out_dir / "pareto_candidates_with_objectives.csv")
    assert not front["reference"].fillna(False).astype(bool).any()
    assert not with_obj["reference"].fillna(False).astype(bool).any()
    assert len(with_obj) == 2
