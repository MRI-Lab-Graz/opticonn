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


def _fake_derivatives(root: Path, subjects: list[str], atlas: str = "AtlasX") -> Path:
    """Minimal qsirecon-shaped tree `scripts.mrtrix_discover_bundle` can resolve."""
    deriv = root / "derivatives"
    qsirecon = deriv / "qsirecon"
    for subject in subjects:
        wf_dwi = qsirecon / "derivatives" / "qsirecon-MRtrix3_act-HSVS" / subject / "dwi"
        wf_dwi.mkdir(parents=True, exist_ok=True)
        (wf_dwi / f"{subject}_label-WM_dwimap.mif.gz").write_text("x")
        dwi = qsirecon / subject / "dwi"
        dwi.mkdir(parents=True, exist_ok=True)
        (dwi / f"{subject}_space-ACPC_seg-{atlas}_dseg.mif.gz").write_text("x")
        (dwi / f"{subject}_seg-{atlas}_dseg.txt").write_text("x")
        anat = qsirecon / subject / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        (anat / f"{subject}_space-ACPC_seg-hsvs_probseg.nii.gz").write_text("x")
    return deriv


def test_tune_grid_mrtrix_forwards_atlas_for_multi_subject_discovery(tmp_path) -> None:
    deriv = _fake_derivatives(tmp_path, ["sub-01", "sub-02"])
    proc = _run(
        [
            "--backend", "mrtrix", "--dry-run", "tune-grid",
            "-i", str(deriv),
            "-o", str(tmp_path / "out"),
            "--subject", "sub-01", "sub-02",
            "--atlas", "AtlasX",
        ]
    )
    combined = proc.stdout + proc.stderr
    assert "--atlas is required" not in combined, combined
    assert proc.returncode == 0, combined


def test_tune_bayes_mrtrix_forwards_atlas_for_multi_subject_discovery(tmp_path) -> None:
    deriv = _fake_derivatives(tmp_path, ["sub-01", "sub-02"])
    proc = _run(
        [
            "--backend", "mrtrix", "--dry-run", "tune-bayes",
            "-i", str(deriv),
            "-o", str(tmp_path / "out"),
            "--subject", "sub-01", "sub-02",
            "--atlas", "AtlasX",
        ]
    )
    combined = proc.stdout + proc.stderr
    assert "--atlas is required" not in combined, combined
    assert proc.returncode == 0, combined


def test_apply_mrtrix_forwards_atlas_for_discovery(tmp_path) -> None:
    deriv = _fake_derivatives(tmp_path, ["sub-01"])
    optimal_config = tmp_path / "optimal.json"
    optimal_config.write_text(json.dumps({"best_parameters": {}}))
    proc = _run(
        [
            "--backend", "mrtrix", "--dry-run", "apply",
            "-i", str(deriv),
            "--optimal-config", str(optimal_config),
            "-o", str(tmp_path / "out"),
            "--subject", "sub-01",
            "--atlas", "AtlasX",
        ]
    )
    combined = proc.stdout + proc.stderr
    assert "Running MRtrix application:" in combined, combined
    assert "--atlas AtlasX" in combined, combined


def test_dsi_tune_grid_forwards_dry_run(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    proc = _run(
        [
            "--dry-run", "tune-grid",
            "-i", str(data_dir),
            "-o", str(tmp_path / "out"),
            "--no-validation",
        ]
    )
    combined = proc.stdout + proc.stderr
    assert "cross_validation_bootstrap_optimizer.py" in combined, combined
    assert "--dry-run" in combined, combined


def _optimizer_cmd(tmp_path, extra: list[str]) -> str:
    """The `Running: ...` line the DSI tune-grid hub prints (--dry-run: no tracking)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sub-1_ses-1.odf.qsdr.fz").write_bytes(b"")
    proc = _run(
        ["--dry-run", "tune-grid", "--no-validation", "-i", str(data_dir), "-o", str(tmp_path / "out"), *extra]
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(" Running:")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return lines[0]


def test_tune_grid_rejects_removed_sessions_per_subject_flag(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    proc = _run(
        ["--dry-run", "tune-grid", "--no-validation", "-i", str(data_dir),
         "-o", str(tmp_path / "out"), "--sessions-per-subject", "2"]
    )
    assert proc.returncode != 0
    assert "unrecognized arguments" in proc.stderr


def test_tune_grid_forwards_ten_subjects_by_default(tmp_path) -> None:
    assert "--subjects 10" in _optimizer_cmd(tmp_path, [])


def test_tune_grid_quick_defaults_to_three_subjects(tmp_path) -> None:
    assert "--subjects 3" in _optimizer_cmd(tmp_path, ["--quick"])


def test_tune_grid_forwards_explicit_subjects(tmp_path) -> None:
    assert "--subjects 4" in _optimizer_cmd(tmp_path, ["--quick", "--subjects", "4"])


def test_tune_grid_rejects_subjects_below_one(tmp_path) -> None:
    proc = _run(["tune-grid", "-i", str(tmp_path), "-o", str(tmp_path / "out"), "--subjects", "0"])
    assert proc.returncode != 0
    assert "--subjects must be at least 1" in proc.stdout + proc.stderr
