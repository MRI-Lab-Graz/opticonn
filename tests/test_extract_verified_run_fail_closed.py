"""Fix round 1 findings on top of test_extract_verified_run.py:

Critical: an exception raised *during* verification (a truncated tract file, an
unreadable/malformed preflight file) must not leave matrices on disk with no
execution record -- it is itself proof the run cannot be trusted. Cleanup must
also be robust to a single file refusing to delete.

Important: a run whose atlases all failed verification must make the CLI
process exit non-zero, in both single-file and batch mode.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.extract_connectivity_matrices import (
    ConnectivityExtractor,
    _has_verification_failure,
    main,
)
from tests.test_extract_verified_run import CONFIG, FIX, _clean_stdout, _run, _assert_failed_closed


def _run_with(tmp_path, side_effect, config=CONFIG):
    ex = ConnectivityExtractor(config)
    with patch("scripts.extract_connectivity_matrices.subprocess.run", side_effect=side_effect):
        result = ex.extract_connectivity_matrix("sub01.qsdr.fz", tmp_path, "AAL3", "sub01")
    atlas_dir = tmp_path / "results" / "AAL3"
    record = json.loads((atlas_dir / "sub01_AAL3.dsi_execution.json").read_text())
    return result, atlas_dir, record


def _fake_dsi_truncated_tract(stdout):
    """subprocess.run stand-in whose tract file is truncated, as a killed DSI
    Studio process would leave it: gzip.decompress() raises EOFError on it."""
    def run(cmd, **_):
        out = Path(next(t.split("=", 1)[1] for t in cmd if t.startswith("--output=")))
        atlas = next(t.split("=", 1)[1] for t in cmd if t.startswith("--connectivity="))
        out.write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 8)  # gzip header, no stream body
        import numpy as np
        import scipy.io
        scipy.io.savemat(str(out) + f".{atlas}.connectivity.mat",
                         {"number of tracts r2r": np.ones((3, 3))})
        (out.parent / (out.name + f".{atlas}.count.connectogram.txt")).write_text("x")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    return run


def test_truncated_tract_file_fails_closed_with_verification_raised(tmp_path):
    result, atlas_dir, record = _run_with(tmp_path, _fake_dsi_truncated_tract(_clean_stdout()))
    _assert_failed_closed(result, atlas_dir, record, "verification raised")


def test_malformed_preflight_fails_closed_with_verification_raised(tmp_path):
    pre = tmp_path / "preflight.json"
    pre.write_text("{not json")
    cfg = {**CONFIG, "verification": {"expected_fingerprints": str(pre)}}
    result, atlas_dir, record = _run(tmp_path, config=cfg, stdout=_clean_stdout())
    _assert_failed_closed(result, atlas_dir, record, "verification raised")


def test_missing_preflight_file_fails_closed_with_verification_raised(tmp_path):
    cfg = {**CONFIG, "verification": {"expected_fingerprints": str(tmp_path / "nope.json")}}
    result, atlas_dir, record = _run(tmp_path, config=cfg, stdout=_clean_stdout())
    _assert_failed_closed(result, atlas_dir, record, "verification raised")


def test_a_failing_unlink_does_not_stop_the_rest_of_cleanup(tmp_path):
    """A file that refuses to delete (e.g. still held open) must not leave the
    other, deletable outputs of a failed run behind, and must be recorded."""
    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **kw):
        if self.name.endswith(".connectivity.mat"):
            raise OSError("disk full")
        return real_unlink(self, *a, **kw)

    with patch.object(Path, "unlink", flaky_unlink):
        result, atlas_dir, record = _run(
            tmp_path, stdout=(FIX / "sweep_0001_stdout.txt").read_text()
        )

    assert result["success"] is False
    assert record["verified"] is False
    left = sorted(p.name for p in atlas_dir.iterdir())
    # the .mat that refused to delete is still there; everything else deletable is gone
    assert left == [
        "sub01_AAL3.dsi_command.txt",
        "sub01_AAL3.dsi_execution.json",
        "sub01_AAL3.tt.gz.AAL3.connectivity.mat",
    ], left
    assert any("could not delete" in e and "connectivity.mat" in e
               for e in result["verification_errors"]), result["verification_errors"]
    assert any("could not delete" in e for e in record["errors"]), record["errors"]


def test_has_verification_failure_true_when_any_atlas_failed():
    assert _has_verification_failure(
        [{"verification_errors": []}, {"verification_errors": ["boom"]}]
    ) is True


def test_has_verification_failure_false_when_all_clean():
    assert _has_verification_failure(
        [{"verification_errors": []}, {"verification_errors": []}]
    ) is False


def test_main_single_file_exits_nonzero_on_verification_failure(tmp_path, monkeypatch):
    input_file = tmp_path / "sub01.fz"
    input_file.write_bytes(b"x")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr("sys.argv", ["prog", str(input_file), str(output_dir)])

    fake_summary = {"results": [{"atlas": "AAL3", "success": False,
                                 "verification_errors": ["boom"]}]}

    with patch.object(ConnectivityExtractor, "validate_configuration",
                      return_value={"valid": True, "warnings": []}), \
         patch.object(ConnectivityExtractor, "validate_input_path",
                      return_value={"valid": True, "files_found": [str(input_file)]}), \
         patch.object(ConnectivityExtractor, "check_dsi_studio",
                      return_value={"path": "x", "version": "1", "available": True}), \
         patch.object(ConnectivityExtractor, "extract_all_matrices",
                      return_value=fake_summary):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0


def test_main_batch_exits_nonzero_on_verification_failure(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    fiber_file = input_dir / "sub01.fz"
    fiber_file.write_bytes(b"x")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr("sys.argv", ["prog", "--batch", str(input_dir), str(output_dir)])

    fake_summary = {"results": [{"atlas": "AAL3", "success": False,
                                 "verification_errors": ["boom"]}],
                     "output_folder": str(output_dir), "matrices_extracted": 0}

    with patch.object(ConnectivityExtractor, "validate_configuration",
                      return_value={"valid": True, "warnings": []}), \
         patch.object(ConnectivityExtractor, "validate_input_path",
                      return_value={"valid": True, "files_found": [str(fiber_file)]}), \
         patch.object(ConnectivityExtractor, "check_dsi_studio",
                      return_value={"path": "x", "version": "1", "available": True}), \
         patch.object(ConnectivityExtractor, "extract_all_matrices",
                      return_value=fake_summary):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0
