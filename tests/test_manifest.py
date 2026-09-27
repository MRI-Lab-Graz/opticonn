import hashlib
import json
from pathlib import Path

import pytest

from scripts import manifest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_write_then_verify_round_trips_clean(tmp_path):
    f = tmp_path / "sub-01.qsdr.fz"
    f.write_bytes(b"fake fib content")
    manifest.write_manifest(tmp_path, stage="fetch",
                            outputs={"sub-01.qsdr.fz": f},
                            contract={"hub_release_tag": "1.0.0"})
    problems = manifest.verify_manifest(tmp_path, {"hub_release_tag": "1.0.0"})
    assert problems == []


def test_verify_fails_closed_on_a_missing_manifest(tmp_path):
    problems = manifest.verify_manifest(tmp_path, {"hub_release_tag": "1.0.0"})
    assert problems == [f"manifest.json is missing in {tmp_path}"]


def test_verify_fails_closed_on_invalid_json(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    problems = manifest.verify_manifest(tmp_path, {"k": "v"})
    assert problems == [f"manifest.json in {tmp_path} is not valid JSON"]


def test_verify_fails_closed_on_a_tampered_file(tmp_path):
    f = tmp_path / "sub-01.qsdr.fz"
    f.write_bytes(b"original content")
    manifest.write_manifest(tmp_path, stage="fetch", outputs={"sub-01.qsdr.fz": f},
                            contract={})
    f.write_bytes(b"tampered content")
    problems = manifest.verify_manifest(tmp_path, {})
    assert problems == ["sub-01.qsdr.fz does not match its manifest hash"]


def test_verify_fails_closed_on_a_missing_output_file(tmp_path):
    f = tmp_path / "sub-01.qsdr.fz"
    f.write_bytes(b"original content")
    manifest.write_manifest(tmp_path, stage="fetch", outputs={"sub-01.qsdr.fz": f},
                            contract={})
    f.unlink()
    problems = manifest.verify_manifest(tmp_path, {})
    assert problems == ["sub-01.qsdr.fz is missing"]


def test_verify_fails_closed_on_a_contract_mismatch(tmp_path):
    manifest.write_manifest(tmp_path, stage="fetch", outputs={},
                            contract={"hub_release_tag": "1.0.0"})
    problems = manifest.verify_manifest(tmp_path, {"hub_release_tag": "2.0.0"})
    assert problems == ["hub_release_tag is '1.0.0', expected '2.0.0'"]


def test_verify_fails_closed_when_expected_key_is_absent_from_the_manifest(tmp_path):
    manifest.write_manifest(tmp_path, stage="fetch", outputs={}, contract={})
    problems = manifest.verify_manifest(tmp_path, {"hub_release_tag": "1.0.0"})
    assert problems == ["hub_release_tag is '<missing>', expected '1.0.0'"]


def test_verify_checks_flat_json_with_no_contract_or_outputs_keys(tmp_path):
    # A file like OptiConn's own preflight.json: flat, no "contract"/"outputs" nesting.
    (tmp_path / "preflight.json").write_text(json.dumps({
        "passed": True, "dsi_apptainer_image": "dsi_studio_hou-2026-09-27.sif",
    }))
    problems = manifest.verify_manifest(
        tmp_path, {"passed": True, "dsi_apptainer_image": "dsi_studio_hou-2026-09-27.sif"},
        filename="preflight.json")
    assert problems == []
    problems = manifest.verify_manifest(
        tmp_path, {"passed": True, "dsi_apptainer_image": "some_other_build.sif"},
        filename="preflight.json")
    assert problems == [
        "dsi_apptainer_image is 'dsi_studio_hou-2026-09-27.sif', expected 'some_other_build.sif'"
    ]
