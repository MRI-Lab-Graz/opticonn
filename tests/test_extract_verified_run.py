"""A tracking run counts only if DSI Studio provably executed what was sent.

Supersedes test_extract_connectivity_matrix_success_path.py, which guarded a dead
call left in the success branch by an incomplete rename (commit 205a1e5): the
success path must still return without raising.
"""

import gzip
import io
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.io

from scripts.extract_connectivity_matrices import ConnectivityExtractor

FIX = Path(__file__).parent / "fixtures" / "dsi_studio_echo"
WARNING = "--connectivity_threshold is not used/recognized"
# The configuration whose DSI Studio run produced the sweep_0001 / exec_grid fixtures.
CONFIG = {
    "tract_count": 50000, "thread_count": 1, "connectivity_values": ["count", "qa"],
    "tracking_parameters": {"turning_angle": 35, "step_size": 1.0, "smoothing": 0.1,
                            "min_length": 10, "max_length": 250, "random_seed": 1},
}


def _clean_stdout():
    return "\n".join(l for l in (FIX / "sweep_0001_stdout.txt").read_text().splitlines()
                     if WARNING not in l)


def _fake_dsi(stdout, report_name="grid", write_tract=True):
    """subprocess.run stand-in: writes what DSI Studio would, returns its stdout."""
    def run(cmd, **_):
        out = Path(next(t.split("=", 1)[1] for t in cmd if t.startswith("--output=")))
        atlas = next(t.split("=", 1)[1] for t in cmd if t.startswith("--connectivity="))
        if write_tract:
            u8 = lambda s: np.frombuffer(s.encode(), dtype=np.uint8)
            buf = io.BytesIO()
            scipy.io.savemat(buf, {
                "report": u8((FIX / f"exec_{report_name}_report.txt").read_text()),
                "parameter_id": u8((FIX / f"exec_{report_name}_parameter_id.txt").read_text()),
                "track": u8("streamlines")})
            out.write_bytes(gzip.compress(buf.getvalue()))
        scipy.io.savemat(str(out) + f".{atlas}.connectivity.mat",
                         {"number of tracts r2r": np.ones((3, 3))})
        (out.parent / (out.name + f".{atlas}.count.connectogram.txt")).write_text("x")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    return run


def _run(tmp_path, config=CONFIG, **fake):
    ex = ConnectivityExtractor(config)
    with patch("scripts.extract_connectivity_matrices.subprocess.run", side_effect=_fake_dsi(**fake)):
        result = ex.extract_connectivity_matrix("sub01.qsdr.fz", tmp_path, "AAL3", "sub01")
    atlas_dir = tmp_path / "results" / "AAL3"
    record = json.loads((atlas_dir / "sub01_AAL3.dsi_execution.json").read_text())
    return result, atlas_dir, record


def test_verified_run_succeeds_and_is_recorded(tmp_path):
    result, atlas_dir, record = _run(tmp_path, stdout=_clean_stdout())
    assert result["success"] is True
    assert result["verification_errors"] == []
    assert record["verified"] is True
    assert record["executed"]["turning_angle"] == {"kind": "fixed", "value": 35.0}
    assert record["parameter_id"] == (FIX / "exec_grid_parameter_id.txt").read_text().strip()
    assert "--turning_angle=35" in (atlas_dir / "sub01_AAL3.dsi_command.txt").read_text()
    assert list(atlas_dir.glob("*.connectivity.mat"))
    assert not list(atlas_dir.glob("*.tt.gz")), "tract file is deleted once verified"


def _assert_failed_closed(result, atlas_dir, record, fragment):
    assert result["success"] is False
    assert any(fragment in e for e in result["verification_errors"]), result["verification_errors"]
    assert record["verified"] is False
    left = sorted(p.name for p in atlas_dir.iterdir())
    assert left == ["sub01_AAL3.dsi_command.txt", "sub01_AAL3.dsi_execution.json"], left


def test_unrecognised_flag_fails_closed(tmp_path):
    _assert_failed_closed(*_run(tmp_path, stdout=(FIX / "sweep_0001_stdout.txt").read_text()),
                          "connectivity_threshold")


def test_report_contradicting_what_was_sent_fails_closed(tmp_path):
    cfg = {**CONFIG, "tracking_parameters": {**CONFIG["tracking_parameters"],
                                             "otsu_threshold": 0.6, "fa_threshold": 0.0}}
    _assert_failed_closed(*_run(tmp_path, config=cfg, stdout=_clean_stdout(),
                                report_name="otsu08"), "anisotropy")


def test_missing_tract_file_fails_closed(tmp_path):
    _assert_failed_closed(*_run(tmp_path, stdout=_clean_stdout(), write_tract=False),
                          "tract file")


def test_fingerprint_mismatch_fails_closed(tmp_path):
    pre = tmp_path / "preflight.json"
    pre.write_text(json.dumps({"passed": True, "expected_fingerprints": {}}))
    cfg = {**CONFIG, "verification": {"expected_fingerprints": str(pre)}}
    _assert_failed_closed(*_run(tmp_path, config=cfg, stdout=_clean_stdout()),
                          "preflight fingerprint")


def test_a_failed_preflight_cannot_vouch_for_anything(tmp_path):
    pre = tmp_path / "preflight.json"
    pre.write_text(json.dumps({"passed": False, "expected_fingerprints": {}}))
    cfg = {**CONFIG, "verification": {"expected_fingerprints": str(pre)}}
    _assert_failed_closed(*_run(tmp_path, config=cfg, stdout=_clean_stdout()),
                          "preflight did not pass")


def test_keep_tract_leaves_the_tract_file_for_the_preflight(tmp_path):
    cfg = {**CONFIG, "verification": {"keep_tract": True}}
    _, atlas_dir, _ = _run(tmp_path, config=cfg, stdout=_clean_stdout())
    assert list(atlas_dir.glob("*.tt.gz"))
