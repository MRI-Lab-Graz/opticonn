"""Dry-run integration: run_wave_pipeline stages sessions per subject."""

import json
from pathlib import Path

from scripts.cross_validation_bootstrap_optimizer import (
    generate_single_wave_config,
    generate_wave_configs,
    run_wave_pipeline,
)
from scripts.utils.discovery import find_subject_files, select_scans

# subject -> number of sessions
_LAYOUT = {"001": 3, "002": 1, "003": 2, "004": 1, "005": 3, "006": 2}


def _dataset(root):
    for sub, n in _LAYOUT.items():
        d = root / f"sub-{sub}" / "fib"
        d.mkdir(parents=True)
        for ses in range(1, n + 1):
            (d / f"sub-{sub}_ses-{ses}.odf.qsdr.fz").write_bytes(b"")
    return root


def _run(tmp_path, **selection):
    data = _dataset(tmp_path / "data")
    sel = {"source_dir": str(data), "n_subjects": 2, "random_seed": 42, "file_pattern": "*.fz"}
    sel.update(selection)
    cfg = {"test_config": {"name": "w"}, "data_selection": sel}
    cfg_path = tmp_path / "wave.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    run_wave_pipeline(str(cfg_path), str(out), dry_run=True)
    lines = (out / "w" / "selected_files.txt").read_text().splitlines()
    return data, [Path(x) for x in lines]


def test_session_aware_wave_stages_two_sessions_of_two_multisession_subjects(tmp_path):
    _, got = _run(tmp_path, sessions_per_subject=2)
    assert len(got) == 4
    by_sub = {}
    for p in got:
        by_sub.setdefault(p.name.split("_")[0], []).append(p.name)
    assert len(by_sub) == 2
    assert all(len(v) == 2 for v in by_sub.values())
    assert not set(by_sub) & {"sub-002", "sub-004"}  # single-session subjects


def test_legacy_config_without_key_samples_scans_exactly_as_before(tmp_path):
    data, got = _run(tmp_path)
    pool = find_subject_files(data, ["*.fz"])
    assert got == select_scans(pool, 2, 42, 0)
    assert len(got) == 2


def test_config_generators_write_sessions_per_subject(tmp_path):
    def sel(path):
        return json.loads(Path(path).read_text())["data_selection"]["sessions_per_subject"]

    w1, w2 = generate_wave_configs("d", tmp_path / "a")
    assert sel(w1) == 2 and sel(w2) == 2
    w1, w2 = generate_wave_configs("d", tmp_path / "b", sessions_per_subject=1)
    assert sel(w1) == 1 and sel(w2) == 1
    assert sel(generate_single_wave_config("d", tmp_path / "c")) == 2
    assert sel(generate_single_wave_config("d", tmp_path / "d", sessions_per_subject=3)) == 3


def test_staging_warns_when_two_selected_scans_share_a_file_name(tmp_path, caplog):
    data = tmp_path / "data"
    for sub in ("007", "008"):
        for ses in (1, 2):
            d = data / f"sub-{sub}" / f"ses-{ses}"
            d.mkdir(parents=True)
            (d / f"sub-{sub}.odf.qsdr.fz").write_bytes(b"")
    cfg = {
        "test_config": {"name": "w"},
        "data_selection": {
            "source_dir": str(data), "n_subjects": 2, "random_seed": 42,
            "file_pattern": "*.fz", "sessions_per_subject": 2,
        },
    }
    cfg_path = tmp_path / "wave.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    with caplog.at_level("WARNING"):
        run_wave_pipeline(str(cfg_path), str(out), dry_run=True)
    listed = (out / "w" / "selected_files.txt").read_text().splitlines()
    assert len(listed) == 4
    assert len(list((out / "w" / "selected_data").iterdir())) == 2
    assert sum("fewer scans than selected" in r.getMessage() for r in caplog.records) == 2


def test_staging_logs_scan_and_subject_counts_in_every_mode(tmp_path, caplog):
    with caplog.at_level("INFO"):
        _run(tmp_path)
    assert any("Staged 2 scans (n_subjects=2, sessions_per_subject=0)" in r.getMessage() for r in caplog.records)
