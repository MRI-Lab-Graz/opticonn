import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTICONN = REPO_ROOT / "opticonn.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(OPTICONN), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_backend_flag_defaults_to_dsi() -> None:
    # The parser is built inline inside main(); no separable builder function
    # exists, so this is the subprocess/--help fallback explicitly allowed by
    # the task brief.
    proc = _run(["--help"])
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "--backend" in proc.stdout
    assert "default: dsi" in proc.stdout


def test_backend_mrtrix_dispatches_to_mrtrix_tune(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    proc = _run(
        [
            "--backend",
            "mrtrix",
            "--dry-run",
            "tune-grid",
            "-i",
            str(data_dir),
            "-o",
            str(out_dir),
        ]
    )

    combined = proc.stdout + proc.stderr
    assert "mrtrix_tune.py" in combined
    assert "sweep" in combined
    # Must return before reaching any DSI-only codepath.
    assert "Full setup validation" not in combined


def test_mrtrix_discover_subcommand_exists_and_requires_atlas(tmp_path) -> None:
    help_proc = _run(["mrtrix-discover", "--help"])
    assert help_proc.returncode == 0, help_proc.stdout + "\n" + help_proc.stderr
    assert "--atlas" in help_proc.stdout

    missing_atlas_proc = _run(
        ["mrtrix-discover", "--qsirecon-dir", str(tmp_path)]
    )
    assert missing_atlas_proc.returncode != 0


def test_apply_backend_mrtrix_before_subcommand_dispatches_to_mrtrix_tune(
    tmp_path,
) -> None:
    optimal_config = tmp_path / "optimal.json"
    optimal_config.write_text(json.dumps({"foo": "bar"}))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    proc = _run(
        [
            "--backend",
            "mrtrix",
            "--dry-run",
            "apply",
            "-i",
            str(data_dir),
            "--optimal-config",
            str(optimal_config),
            "-o",
            str(out_dir),
        ]
    )

    combined = proc.stdout + proc.stderr
    assert "mrtrix_tune.py" in combined
    assert "apply" in combined


def test_apply_backend_mrtrix_after_subcommand_dispatches_to_mrtrix_tune(
    tmp_path,
) -> None:
    optimal_config = tmp_path / "optimal.json"
    optimal_config.write_text(json.dumps({"foo": "bar"}))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    proc = _run(
        [
            "--dry-run",
            "apply",
            "--backend",
            "mrtrix",
            "-i",
            str(data_dir),
            "--optimal-config",
            str(optimal_config),
            "-o",
            str(out_dir),
        ]
    )

    combined = proc.stdout + proc.stderr
    assert "mrtrix_tune.py" in combined
    assert "apply" in combined


def test_apply_without_any_backend_flag_keeps_existing_auto_detect_behavior(
    tmp_path,
) -> None:
    # No run_metadata.backend=="mrtrix" marker anywhere in this config, and
    # --backend never passed (neither before nor after the subcommand) --
    # today's auto-detect-then-default-to-dsi behavior must be unaffected.
    optimal_config = tmp_path / "optimal.json"
    optimal_config.write_text(json.dumps({"foo": "bar"}))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    proc = _run(
        [
            "apply",
            "-i",
            str(data_dir),
            "--optimal-config",
            str(optimal_config),
            "-o",
            str(out_dir),
        ]
    )

    combined = proc.stdout + proc.stderr
    assert "mrtrix_tune.py" not in combined
    assert "Running MRtrix" not in combined


def test_backend_mrtrix_verbose_flag_is_not_forwarded_to_mrtrix_tune(
    tmp_path,
) -> None:
    # scripts/mrtrix_tune.py has no --verbose flag; forwarding it makes the
    # dispatched subprocess fail with "unrecognized arguments: --verbose".
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    proc = _run(
        [
            "--backend",
            "mrtrix",
            "--dry-run",
            "tune-grid",
            "-i",
            str(data_dir),
            "-o",
            str(out_dir),
            "--verbose",
        ]
    )

    combined = proc.stdout + proc.stderr
    assert "unrecognized arguments: --verbose" not in combined
    assert "--verbose" not in combined


def test_mrtrix_discover_requires_subject_and_fails_cleanly_at_hub_level(
    tmp_path,
) -> None:
    proc = _run(
        [
            "mrtrix-discover",
            "--qsirecon-dir",
            str(tmp_path),
            "--atlas",
            "desikan",
        ]
    )

    assert proc.returncode != 0
    # Must fail at the hub's own argparse level, never reach the subprocess.
    assert "Running MRtrix bundle discovery" not in proc.stdout
    assert "--subject" in proc.stderr
