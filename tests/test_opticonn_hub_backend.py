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
