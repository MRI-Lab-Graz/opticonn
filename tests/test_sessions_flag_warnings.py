"""--sessions-per-subject is silently ignored when wave configs are supplied; it must warn."""

import json
import sys
from pathlib import Path

import pytest

from scripts import cross_validation_bootstrap_optimizer as cv
from scripts.json_validator import JSONValidator


def _wave(tmp_path, name):
    data = tmp_path / "data" / "sub-001" / "fib"
    data.mkdir(parents=True, exist_ok=True)
    (data / "sub-001_ses-1.odf.qsdr.fz").write_bytes(b"")
    cfg = {
        "test_config": {"name": name},
        "data_selection": {"source_dir": str(tmp_path / "data"), "n_subjects": 1, "file_pattern": "*.fz"},
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(cfg))
    return path


def _main(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(
        sys, "argv",
        ["cv", "-i", str(tmp_path / "data"), "-o", str(tmp_path / "out"), "--dry-run", *extra],
    )
    try:
        cv.main()
    except SystemExit:
        pass


def test_warns_when_flag_given_with_wave_configs(tmp_path, monkeypatch, caplog):
    w1, w2 = _wave(tmp_path, "w1"), _wave(tmp_path, "w2")
    _main(monkeypatch, tmp_path, "--wave1-config", str(w1), "--wave2-config", str(w2),
          "--sessions-per-subject", "3")
    assert "--sessions-per-subject is ignored" in caplog.text


def test_no_warning_when_flag_omitted_with_wave_configs(tmp_path, monkeypatch, caplog):
    w1, w2 = _wave(tmp_path, "w1"), _wave(tmp_path, "w2")
    _main(monkeypatch, tmp_path, "--wave1-config", str(w1), "--wave2-config", str(w2))
    assert "ignored" not in caplog.text


def test_warns_when_master_config_embeds_waves(tmp_path, monkeypatch, caplog):
    w1, w2 = _wave(tmp_path, "w1"), _wave(tmp_path, "w2")
    master = tmp_path / "master.json"
    master.write_text(json.dumps({"wave1_config": str(w1), "wave2_config": str(w2)}))
    _main(monkeypatch, tmp_path, "--config", str(master), "--sessions-per-subject", "1")
    assert "master config embeds wave configs" in caplog.text


@pytest.mark.parametrize("bad", [-1, "2", 1.5, True])
def test_validator_rejects_bad_sessions_per_subject(tmp_path, bad):
    path = _wave(tmp_path, "w")
    cfg = json.loads(path.read_text())
    cfg["data_selection"]["sessions_per_subject"] = bad
    path.write_text(json.dumps(cfg))
    ok, errors = JSONValidator().validate_config(str(path), dry_run=True)
    assert not ok and any("sessions_per_subject" in e for e in errors)


def test_validator_accepts_zero_and_positive(tmp_path):
    path = _wave(tmp_path, "w")
    for good in (0, 1, 2):
        cfg = json.loads(path.read_text())
        cfg["data_selection"]["sessions_per_subject"] = good
        path.write_text(json.dumps(cfg))
        ok, errors = JSONValidator().validate_config(str(path), dry_run=True)
        assert ok, errors
