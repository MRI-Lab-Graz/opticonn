import json
import subprocess
from pathlib import Path

import nibabel.streamlines
import pytest

from scripts import dsi_preflight as pf

SWEEP = {
    "tract_count": 50000, "thread_count": 1, "connectivity_values": ["count"],
    "reliability": {"repeats": 2},
    "tracking_parameters": {"step_size": 1.0, "smoothing": 0.1, "max_length": 250},
    "sweep_parameters": {
        "fa_threshold_range": [0.0, 0.1], "otsu_range": [0.6, 0.8],
        "turning_angle_range": [35, 50],
        "reference_candidate": {"fa_threshold": None, "turning_angle": None,
                                "step_size": None, "smoothing": None, "max_length": None},
    },
}


def test_enumerates_every_distinct_spec_and_repeat():
    specs = pf.enumerate_specs(SWEEP)
    # (fa 0/otsu 0.6, fa 0/otsu 0.8, fa 0.1) x 2 angles = 6 grid + 1 reference, x 2 repeats
    assert len(specs) == 14
    assert {s["repeat"] for s in specs} == {1, 2}
    assert all(s["config"]["tracking_parameters"]["random_seed"] == s["repeat"] for s in specs)


def test_reference_spec_carries_nulls_through_to_its_config():
    ref = [s for s in pf.enumerate_specs(SWEEP) if s["choice"].get("turning_angle") is None]
    assert ref and all(s["config"]["tracking_parameters"]["turning_angle"] is None for s in ref)


def test_preflight_requires_single_threaded_determinism():
    with pytest.raises(ValueError, match="thread_count"):
        pf.enumerate_specs({**SWEEP, "thread_count": 4})


def test_single_axis_pairs():
    choices = [{"a": 1, "b": 1}, {"a": 2, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 2}]
    assert sorted(pf.single_axis_pairs(choices)) == [
        (0, 1, "a"), (0, 2, "b"), (1, 3, "b"), (2, 3, "a")]


def _run(spec, repeat, key, pid="PID", errors=()):
    return {"spec": spec, "repeat": repeat, "fingerprint_key": key, "parameter_id": pid,
            "errors": list(errors)}


def test_summary_passes_only_when_everything_passed():
    runs = [_run(0, 1, "k0", "p0"), _run(1, 1, "k1", "p1")]
    s = pf.summarize(runs, [(0, 1, "a")], {0: "d0", 1: "d1"})
    assert s["passed"] is True
    assert s["expected_fingerprints"] == {"k0": "p0", "k1": "p1"}


def test_any_run_error_fails_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", errors=["boom"])], [], {0: "d0"})
    assert s["passed"] is False and "boom" in s["failures"][0]


def test_an_axis_that_changes_nothing_fails_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", "p0"), _run(1, 1, "k1", "p1")],
                     [(0, 1, "turning_angle")], {0: "same", 1: "same"})
    assert s["passed"] is False
    assert "turning_angle" in s["failures"][0] and "did not change" in s["failures"][0]


def test_two_specs_with_one_fingerprint_fail_the_preflight():
    s = pf.summarize([_run(0, 1, "k0", "p"), _run(1, 1, "k1", "p")], [], {0: "a", 1: "b"})
    assert s["passed"] is False and "same parameter_id" in s["failures"][0]


def _write_execution_json(atlas_dir: Path, base="pf", atlas="AAL3"):
    atlas_dir.mkdir(parents=True, exist_ok=True)
    info = {"sent": {"a": 1}, "executed": {"tract_count": 1},
            "parameter_id": "PID", "track_sha256": "abc"}
    (atlas_dir / f"{base}_{atlas}.dsi_execution.json").write_text(json.dumps(info))
    (atlas_dir / f"{base}_{atlas}.tt.gz").write_bytes(b"tract")
    return info


class _FakeExtractor:
    """Stands in for ConnectivityExtractor: writes the execution record and
    tract file a real run would, without needing DSI Studio."""

    def __init__(self, cfg):
        self.cfg = cfg

    def extract_connectivity_matrix(self, subject, run_dir, atlas, base):
        _write_execution_json(run_dir / "results" / atlas, base, atlas)
        return {"success": True, "verification_errors": []}


def test_stale_direct_export_is_not_used_when_the_fresh_export_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "ConnectivityExtractor", _FakeExtractor)
    monkeypatch.setattr(pf, "build_track_command",
                        lambda cfg, dsi_cmd, subject, output, atlas: ["dsi_studio", "--action=trk"])

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(pf.subprocess, "run", fake_run)

    out = tmp_path / "out"
    run_dir = out / "spec000_rep1"
    run_dir.mkdir(parents=True)
    stale = run_dir / "direct.trk.gz"
    stale.write_bytes(b"stale-data-from-a-previous-run")

    item = {"spec": 0, "repeat": 1, "choice": {}, "config": {}}
    result = pf._run_spec(item, "subject.fz", out)

    assert any("returncode 1" in e for e in result["errors"])
    assert any("missing" in e for e in result["errors"])
    # the stale file must not survive to be read as this run's proof
    assert not stale.exists()


def test_nonzero_conversion_returncode_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "ConnectivityExtractor", _FakeExtractor)

    def fake_build_cmd(cfg, dsi_cmd, subject, output, atlas):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"direct-export")
        return ["dsi_studio", "--action=trk", f"--output={output}"]

    monkeypatch.setattr(pf, "build_track_command", fake_build_cmd)

    def fake_run(cmd, **kwargs):
        if "--action=ana" in cmd:
            return subprocess.CompletedProcess(cmd, returncode=2, stdout="", stderr="bad tract")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pf.subprocess, "run", fake_run)

    out = tmp_path / "out"
    item = {"spec": 1, "repeat": 1, "choice": {}, "config": {}}
    result = pf._run_spec(item, "subject.fz", out)

    assert any("conversion" in e and "2" in e for e in result["errors"])


def test_direct_and_converted_exports_are_loaded_lazily(tmp_path, monkeypatch):
    """Regression: nib.streamlines.load's non-lazy path calls seek(0, SEEK_END)
    purely to guess a buffer size, and indexed_gzip's IndexedGzipFile (which
    nibabel picks for .gz files when the package is installed) raises
    NotCoveredError on that seek unless its full index is already built --
    reproduced on every real DSI Studio .trk.gz in the 2026-09-27 demo run.
    lazy_load=True must be passed to both load calls to avoid it."""
    monkeypatch.setattr(pf, "ConnectivityExtractor", _FakeExtractor)

    def fake_build_cmd(cfg, dsi_cmd, subject, output, atlas):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"direct-export")
        return ["dsi_studio", "--action=trk", f"--output={output}"]

    monkeypatch.setattr(pf, "build_track_command", fake_build_cmd)

    def fake_run(cmd, **kwargs):
        if "--action=ana" in cmd:
            output = next(a for a in cmd if a.startswith("--output=")).split("=", 1)[1]
            Path(output).write_bytes(b"converted-export")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pf.subprocess, "run", fake_run)

    class _FakeTractogram:
        header = {"voxel_sizes": [1.0, 1.0, 1.0]}
        streamlines: list = []

    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return _FakeTractogram()

    monkeypatch.setattr(nibabel.streamlines, "load", fake_load)
    monkeypatch.setattr(pf.dsi_verify, "same_streamlines", lambda a, b: [])
    monkeypatch.setattr(pf.dsi_verify, "check_geometry", lambda *a, **k: [])

    out = tmp_path / "out"
    item = {"spec": 0, "repeat": 1, "choice": {}, "config": {}}
    result = pf._run_spec(item, "subject.fz", out)

    assert result["errors"] == []
    assert len(calls) == 2
    assert all(kwargs.get("lazy_load") is True for kwargs in calls)


def test_stale_preflight_json_does_not_survive_a_crashing_run(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(SWEEP))
    out = tmp_path / "out"
    out.mkdir()
    (out / "preflight.json").write_text(json.dumps({"passed": True, "failures": []}))

    class _RaisingPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, *a, **k):
            raise FileNotFoundError("dsi_execution.json missing")

    monkeypatch.setattr(pf, "ProcessPoolExecutor", _RaisingPool)

    with pytest.raises(FileNotFoundError):
        pf.main(["--config", str(cfg_path), "--subject", "sub.fz", "--out", str(out)])

    assert not (out / "preflight.json").exists()
