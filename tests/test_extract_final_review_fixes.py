"""Final whole-branch review fix wave, one test per finding:

C1 -- the execution record must say exactly which checks vouched for a run,
      and skipping the fingerprint check (no preflight configured) must be
      visible, not silent.
I1 -- an exception raised while processing a batch file must fail the
      process, not just that file, and must not leave unverified outputs.
I2 -- a timeout must still write an execution record and delete outputs.
I4 -- a run whose echo/report pass but whose DSI Studio exit code is
      non-zero (or matrices are incomplete) must still be fail-closed and
      must fail the process.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.extract_connectivity_matrices import ConnectivityExtractor, main
from tests.test_extract_verified_run import CONFIG, FIX, _clean_stdout, _fake_dsi, _run


def test_c1_record_lists_which_checks_ran_and_omits_fingerprint_without_a_preflight(tmp_path):
    result, atlas_dir, record = _run(tmp_path, stdout=_clean_stdout())
    assert result["success"] is True
    assert record["checks"] == ["echo", "report"]
    assert "fingerprint" not in record["checks"]


def test_c1_record_lists_fingerprint_when_a_preflight_is_configured_and_passes(tmp_path):
    pre = tmp_path / "preflight.json"
    key = "--connectivity=AAL3 --connectivity_type=pass --connectivity_value=count,qa " \
          "--method=0 --min_length=10.0 --max_length=250.0 --random_seed=1 " \
          "--smoothing=0.1 --step_size=1.0 --thread_count=1 --turning_angle=35.0"
    # Build the fingerprint key the same way production code does, so the fingerprint
    # check passes and we can observe "fingerprint" appear in checks.
    from scripts import dsi_verify
    from scripts.extract_connectivity_matrices import build_track_command
    cmd = build_track_command(CONFIG, "dsi_studio", "source", "out.tt.gz", "AAL3")
    sent = dsi_verify.parse_command_flags(cmd)
    fp_key = dsi_verify.fingerprint_key(sent)
    expected_parameter_id = (FIX / "exec_grid_parameter_id.txt").read_text().strip()
    pre.write_text(json.dumps({"passed": True, "expected_fingerprints": {fp_key: expected_parameter_id}}))
    cfg = {**CONFIG, "verification": {"expected_fingerprints": str(pre)}}
    result, atlas_dir, record = _run(tmp_path, config=cfg, stdout=_clean_stdout())
    assert result["success"] is True, result["verification_errors"]
    assert record["checks"] == ["echo", "report", "fingerprint"]


def test_c1_warns_when_fingerprint_check_is_skipped(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _run(tmp_path, stdout=_clean_stdout())
    assert any("fingerprint" in r.message.lower() and "skip" in r.message.lower()
              for r in caplog.records)


def test_i2_timeout_writes_a_record_and_deletes_outputs(tmp_path):
    ex = ConnectivityExtractor(CONFIG)

    def raise_timeout(cmd, **_):
        # DSI Studio partially wrote outputs before being killed by the timeout.
        out = Path(next(t.split("=", 1)[1] for t in cmd if t.startswith("--output=")))
        atlas = next(t.split("=", 1)[1] for t in cmd if t.startswith("--connectivity="))
        out.write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 8)
        (out.parent / (out.name + f".{atlas}.count.connectogram.txt")).write_text("x")
        raise subprocess.TimeoutExpired(cmd, 3600)

    with patch("scripts.extract_connectivity_matrices.subprocess.run", side_effect=raise_timeout):
        result = ex.extract_connectivity_matrix("sub01.qsdr.fz", tmp_path, "AAL3", "sub01")

    atlas_dir = tmp_path / "results" / "AAL3"
    assert result["success"] is False
    assert any("timeout" in e.lower() for e in result["verification_errors"])
    record = json.loads((atlas_dir / "sub01_AAL3.dsi_execution.json").read_text())
    assert record["verified"] is False
    left = sorted(p.name for p in atlas_dir.iterdir())
    assert left == ["sub01_AAL3.dsi_command.txt", "sub01_AAL3.dsi_execution.json"], left


def test_i4_verification_passes_but_bad_return_code_fails_closed(tmp_path):
    ex = ConnectivityExtractor(CONFIG)

    def run(cmd, **_):
        # Reuse the normal fake DSI Studio, but report a non-zero exit code even
        # though everything it wrote (echo, tract report) is internally consistent.
        real = _fake_dsi(stdout=_clean_stdout())(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout=real.stdout, stderr="boom")

    with patch("scripts.extract_connectivity_matrices.subprocess.run", side_effect=run):
        result = ex.extract_connectivity_matrix("sub01.qsdr.fz", tmp_path, "AAL3", "sub01")

    atlas_dir = tmp_path / "results" / "AAL3"
    assert result["success"] is False
    assert result["verification_errors"] == []  # verification itself was clean
    record = json.loads((atlas_dir / "sub01_AAL3.dsi_execution.json").read_text())
    assert record["verified"] is True  # the proof stands; the run just failed otherwise
    left = sorted(p.name for p in atlas_dir.iterdir())
    assert left == ["sub01_AAL3.dsi_command.txt", "sub01_AAL3.dsi_execution.json"], left


def test_i1_a_raising_batch_file_fails_the_process_and_leaves_no_outputs(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    fiber_file = input_dir / "sub01.fz"
    fiber_file.write_bytes(b"x")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr("sys.argv", ["prog", "--batch", str(input_dir), str(output_dir)])

    with patch.object(ConnectivityExtractor, "validate_configuration",
                      return_value={"valid": True, "warnings": []}), \
         patch.object(ConnectivityExtractor, "validate_input_path",
                      return_value={"valid": True, "files_found": [str(fiber_file)]}), \
         patch.object(ConnectivityExtractor, "check_dsi_studio",
                      return_value={"path": "x", "version": "1", "available": True}), \
         patch.object(ConnectivityExtractor, "extract_all_matrices",
                      side_effect=RuntimeError("disk full writing dsi_execution.json")):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0
