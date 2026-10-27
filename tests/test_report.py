import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.report as report_module
from scripts.report import build_report, flat_row
from tests.sweep_fixture import make_sweep

ROOT = Path(__file__).resolve().parents[1]
SPECS = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 0), ({"fa_threshold": 0.2}, False, 0),
         ({"fa_threshold": 0.3}, False, 50),                    # rejected by the (fixture's) gate
         ({"fa_threshold": 0.4}, False, 0), ({"fa_threshold": None}, True, 0)]


def _two_waves(root, repeats=2):
    make_sweep(root, SPECS, n_subjects=6, wave="wave1", repeats=repeats, margin=lambda i: 0.05 * i)
    make_sweep(root, SPECS, n_subjects=6, wave="wave2", repeats=repeats, margin=lambda i: 0.05 * i + 0.001)
    return root


def test_a_full_sweep_gets_every_statistic(tmp_path):
    rep = build_report(_two_waves(tmp_path / "opt"), workers=1)
    assert rep["sweep"]["waves"] == ["wave1", "wave2"] and rep["sweep"]["n_specifications"] == 12
    assert rep["sweep"]["n_rejected"] == 2 and rep["sweep"]["reference"] == ["wave1/sweep_0006", "wave2/sweep_0006"]
    lead = rep["leader"]
    assert lead["n_candidates"] == 4 and lead["winners_agree"] is True and lead["winner_determined"] is True
    assert lead["cross_wave_rho"] == pytest.approx(1.0)
    assert rep["fragility"]["n_seed_pairs"] > 0 and rep["fragility"]["seed_rho"] <= 1.0
    assert rep["effective_dimensionality"]["n_ed_rows"] > 0
    assert rep["displacement"]["available"] is True and rep["reference_rho"]["n_reference_ordering_pairs"] > 0
    assert rep["density_relation"]["reference_rejected"] is False
    assert 0.0 <= rep["density_relation"]["reference_density_percentile"] <= 1.0
    assert "small_worldness(binary)_vs_density" in rep["density_relation"]["relations"]
    assert rep["notes"] == []


def test_the_rejected_specifications_do_not_enter_the_pooled_statistics(tmp_path):
    rep = build_report(_two_waves(tmp_path / "opt"), workers=1)
    # 12 specifications, 2 rejected: fragility seed arm = measures x kept specifications
    assert rep["fragility"]["n_seed_pairs"] == rep["fragility"]["n_measures"] * 10


def test_report_files_are_written_and_the_json_is_strict(tmp_path):
    opt = _two_waves(tmp_path / "opt")
    build_report(opt, workers=1)
    text = (opt / "report" / "report.json").read_text()
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-standard {c}")))
    with open(opt / "report" / "report.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]["n_candidates"] == "4" and rows[0]["winner_determined"] == "True"
    assert "seed_rho" in rows[0] and "reference_over_between_subject" in rows[0]


def test_a_single_wave_single_repeat_sweep_skips_what_it_cannot_do_and_says_why(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=6, repeats=1, margin=lambda i: 0.05 * i)
    rep = build_report(tmp_path / "opt", workers=1)
    assert rep["leader"] is None and rep["fragility"] is None and rep["effective_dimensionality"] is None
    assert rep["displacement"]["available"] is True and rep["density_relation"] is not None
    notes = " ".join(rep["notes"])
    assert "two waves" in notes and "two repeats" in notes
    assert flat_row(rep)["n_candidates"] is None and "n_candidates" in flat_row(rep)


def test_a_sweep_without_a_reference_skips_displacement_and_reference_statistics(tmp_path):
    specs = [s for s in SPECS if not s[1]]
    make_sweep(tmp_path / "opt", specs, n_subjects=6, repeats=2, margin=lambda i: 0.05 * i)
    rep = build_report(tmp_path / "opt", workers=1)
    assert rep["displacement"] is None and rep["reference_rho"] is None and "reference" in " ".join(rep["notes"])
    assert rep["density_relation"]["reference_density_percentile"] is None


def test_existing_tables_are_reused_until_recompute_is_asked(tmp_path, monkeypatch):
    calls = []
    real = report_module.write_measures
    monkeypatch.setattr(report_module, "write_measures", lambda *a, **k: (calls.append(k["repeat"]), real(*a, **k))[1])
    opt = _two_waves(tmp_path / "opt")
    build_report(opt, workers=1)
    assert calls == [1, 2]
    build_report(opt, workers=1)
    assert calls == [1, 2]                                   # cached
    build_report(opt, workers=1, recompute=True)
    assert calls == [1, 2, 1, 2]


def test_an_ambiguous_metric_is_a_clean_error_and_the_command_runs_through_the_hub(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=6, repeats=2, metrics=("count", "qa"), margin=lambda i: 0.05 * i)
    with pytest.raises(SystemExit, match="more than one atlas/metric"):
        report_module.main(["-i", str(tmp_path / "opt"), "--workers", "1"])
    out = subprocess.run([sys.executable, str(ROOT / "opticonn.py"), "report", "-i", str(tmp_path / "opt"),
                          "--metric", "count", "--workers", "1"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "opt" / "report" / "report.json").exists()
    help_out = subprocess.run([sys.executable, str(ROOT / "opticonn.py"), "report", "--help"], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)
    assert help_out.returncode == 0 and "--recompute" in help_out.stdout


def test_a_corrupt_repeat_two_table_is_a_clean_error_not_a_missing_repeat(tmp_path):
    opt = _two_waves(tmp_path / "opt")
    build_report(opt, workers=1)
    path = opt / "report" / "tables" / "repeat2" / "measures.csv"
    lines = path.read_text().splitlines()
    head = lines[0].split(",")
    first = lines[1].split(",")
    first[head.index("density")] = "oops"
    path.write_text("\n".join([lines[0], ",".join(first), *lines[2:]]) + "\n")
    with pytest.raises(ValueError, match="could not convert"):
        build_report(opt, workers=1)
    with pytest.raises(SystemExit, match="opticonn report: .*could not convert"):
        report_module.main(["-i", str(opt), "--workers", "1"])


def test_a_missing_second_repeat_is_a_note_and_never_cached_or_attempted(tmp_path, monkeypatch):
    calls = []
    real = report_module.write_measures
    monkeypatch.setattr(report_module, "write_measures", lambda *a, **k: (calls.append(k["repeat"]), real(*a, **k))[1])
    make_sweep(tmp_path / "opt", SPECS, n_subjects=6, repeats=1, margin=lambda i: 0.05 * i)
    for _ in range(2):
        rep = build_report(tmp_path / "opt", workers=1)
        assert "two repeats" in " ".join(rep["notes"])
    assert calls == [1]


def test_tables_are_rebuilt_when_the_sweep_gains_subjects(tmp_path, monkeypatch):
    calls = []
    real = report_module.write_measures
    monkeypatch.setattr(report_module, "write_measures", lambda *a, **k: (calls.append(k["repeat"]), real(*a, **k))[1])
    opt = tmp_path / "opt"
    make_sweep(opt, SPECS, n_subjects=6, repeats=2, margin=lambda i: 0.05 * i)
    assert build_report(opt, workers=1)["effective_dimensionality"]["n_ed_subjects"] == 6
    make_sweep(opt, SPECS, n_subjects=8, repeats=2, margin=lambda i: 0.05 * i)
    rep = build_report(opt, workers=1)
    assert calls == [1, 2, 1, 2] and rep["effective_dimensionality"]["n_ed_subjects"] == 8


def test_a_wave_without_diagnostics_is_noted_and_not_gated(tmp_path):
    opt = _two_waves(tmp_path / "opt")
    (opt / "wave2" / "combo_diagnostics.csv").unlink()
    rep = build_report(opt, workers=1)
    assert any(n.startswith("wave2: no combo_diagnostics.csv") and "not gated" in n for n in rep["notes"])
    assert rep["sweep"]["n_rejected"] == 1


def test_three_waves_skip_the_leader_test_but_keep_within_wave_statistics(tmp_path):
    one = build_report(make_sweep(tmp_path / "one", SPECS, n_subjects=6, repeats=2, margin=lambda i: 0.05 * i), workers=1)
    opt = tmp_path / "opt"
    for k in (1, 2, 3):
        make_sweep(opt, SPECS, n_subjects=6, wave=f"wave{k}", repeats=2, margin=lambda i: 0.05 * i + 0.001 * k)
    rep = build_report(opt, workers=1)
    assert rep["leader"] is None and any("exactly two waves" in n for n in rep["notes"])
    assert rep["sweep"]["waves"] == ["wave1", "wave2", "wave3"]
    assert rep["fragility"]["n_param_pairs"] == 3 * one["fragility"]["n_param_pairs"] > 0
    assert rep["displacement"]["available"] is True


def test_a_rejected_reference_gets_one_clear_note(tmp_path):
    specs = [s for s in SPECS if not s[1] and s[2] == 0] + [({"fa_threshold": None}, True, 50)]
    rep = build_report(make_sweep(tmp_path / "opt", specs, n_subjects=6, repeats=2, margin=lambda i: 0.05 * i), workers=1)
    assert not (rep["displacement"] or {}).get("available") and rep["density_relation"]["reference_rejected"] is True
    mine = [n for n in rep["notes"] if "reference was rejected by the gate (sweep_0005)" in n]
    assert len(mine) == 1 and not any("same wave" in n for n in rep["notes"])
    assert len(rep["notes"]) == len(set(rep["notes"]))


def test_the_csv_columns_are_fixed_whatever_the_sweep(tmp_path):
    def header(name, specs):
        opt = make_sweep(tmp_path / name, specs, n_subjects=6, repeats=2, margin=lambda i: 0.05 * i)
        build_report(opt, workers=1)
        with open(opt / "report" / "report.csv") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames, next(reader)
    spread = [({"fa_threshold": t}, False, o) for t, o in ((0.0, 0), (0.1, 8), (0.2, 16), (0.3, 24), (0.4, 32))]
    full_head, full_row = header("full", spread + [({"fa_threshold": None}, True, 0)])
    rej_head, _ = header("rej", [({"fa_threshold": 0.0}, False, 50), ({"fa_threshold": 0.1}, False, 50),
                                 ({"fa_threshold": None}, True, 0)])
    assert full_head == rej_head
    assert full_row["small_worldness(binary)_vs_density"] != ""


def test_one_rejected_reference_does_not_claim_the_others_were_skipped(tmp_path):
    kept = [s for s in SPECS if not s[1]]
    opt = tmp_path / "opt"
    make_sweep(opt, kept + [({"fa_threshold": None}, True, 50)], n_subjects=6, wave="wave1", repeats=2, margin=lambda i: 0.05 * i)
    make_sweep(opt, kept + [({"fa_threshold": None}, True, 0)], n_subjects=6, wave="wave2", repeats=2, margin=lambda i: 0.05 * i)
    rep = build_report(opt, workers=1)
    assert rep["displacement"]["available"] is True
    assert not any("skipped: the reference was rejected" in n for n in rep["notes"])
    assert any("wave1/sweep_0006" in n and "rejected" in n for n in rep["notes"])


def test_the_top_closeness_fields_reach_the_report_and_the_csv(tmp_path):
    opt = _two_waves(tmp_path / "opt")
    rep = build_report(opt, workers=1)
    assert rep["leader"]["winner_wave1_rank_in_wave2"] == 1 and rep["leader"]["top5_overlap"] == 4   # 4 candidates
    with open(opt / "report" / "report.csv") as handle:
        row = next(csv.DictReader(handle))
    assert row["winner_wave1_rank_in_wave2"] == "1" and row["top5_overlap"] == "4"
