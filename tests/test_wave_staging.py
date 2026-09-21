"""Dry-run integration: run_wave_pipeline stages one baseline scan per subject."""

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.bayesian_optimizer import BayesianOptimizer
from scripts.cross_validation_bootstrap_optimizer import (
    generate_single_wave_config,
    generate_wave_configs,
    run_wave_pipeline,
)
from scripts.utils.discovery import baseline_scans, find_subject_files, select_scans

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


def test_wave_stages_one_baseline_scan_per_subject(tmp_path):
    _, got = _run(tmp_path, n_subjects=20)
    assert sorted(p.name for p in got) == [f"sub-{s}_ses-1.odf.qsdr.fz" for s in sorted(_LAYOUT)]


def test_wave_staging_is_the_seeded_sample_of_baseline_scans(tmp_path):
    data, got = _run(tmp_path, n_subjects=3)
    pool = baseline_scans(find_subject_files(data, ["*.fz"]))
    assert got == select_scans(pool, 3, 42)
    assert len(got) == 3


def test_excluding_a_baseline_scan_drops_the_subject(tmp_path):
    _, got = _run(tmp_path, n_subjects=20, exclude_scans=["sub-001_ses-1"])
    names = {p.name for p in got}
    assert not any(n.startswith("sub-001_") for n in names)  # no fallback to ses-2
    assert len(names) == 5


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
        d = data / f"sub-{sub}" / "ses-1"
        d.mkdir(parents=True)
        (d / "dwi.odf.qsdr.fz").write_bytes(b"")  # same name, different subjects
    cfg = {
        "test_config": {"name": "w"},
        "data_selection": {
            "source_dir": str(data), "n_subjects": 2, "random_seed": 42, "file_pattern": "*.fz",
        },
    }
    cfg_path = tmp_path / "wave.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    with caplog.at_level("WARNING"):
        run_wave_pipeline(str(cfg_path), str(out), dry_run=True)
    listed = (out / "w" / "selected_files.txt").read_text().splitlines()
    assert len(listed) == 2
    assert len(list((out / "w" / "selected_data").iterdir())) == 1
    assert sum("fewer scans than selected" in r.getMessage() for r in caplog.records) == 1


def test_staging_logs_scan_count_one_per_subject(tmp_path, caplog):
    with caplog.at_level("INFO"):
        _run(tmp_path)
    assert any("Staged 2 scans, one per subject (n_subjects=2)" in r.getMessage() for r in caplog.records)


def test_bayesian_sampling_pool_holds_one_baseline_scan_per_subject(tmp_path):
    data = _dataset(tmp_path / "data")
    got = BayesianOptimizer._get_all_subjects(SimpleNamespace(data_dir=data))
    assert sorted(p.name for p in got) == [f"sub-{s}_ses-1.odf.qsdr.fz" for s in sorted(_LAYOUT)]
