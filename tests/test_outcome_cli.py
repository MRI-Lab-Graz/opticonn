import subprocess
import sys
from pathlib import Path

from scripts.measures_table import write_measures  # noqa: F401  (import check)
from tests.sweep_fixture import make_sweep

ROOT = Path(__file__).resolve().parents[1]
SPECS = [({"fa_threshold": 0.0}, False, 0), ({"fa_threshold": 0.1}, False, 2), ({"fa_threshold": None}, True, 0)]


def _run(*args):
    return subprocess.run([sys.executable, str(ROOT / "opticonn.py"), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)


def test_each_command_has_help_through_the_hub():
    for cmd, flag in (("measures", "--extended"), ("outcome", "--predictor"), ("pca", "--anchor")):
        out = _run(cmd, "--help")
        assert out.returncode == 0, out.stderr
        assert flag in out.stdout


def test_the_three_commands_run_in_sequence_on_a_sweep(tmp_path):
    make_sweep(tmp_path / "opt", SPECS, n_subjects=12)
    (tmp_path / "participants.tsv").write_text(
        "participant_id\tage\tsex\n" + "".join(f"{i + 1:02d}\t{20 + 3 * i}\t{'M' if i % 2 else 'F'}\n" for i in range(12)))
    opt = str(tmp_path / "opt")
    assert _run("measures", "-i", opt, "--workers", "1").returncode == 0
    out = _run("outcome", "-i", opt, "--variables", str(tmp_path / "participants.tsv"), "--predictor", "age",
               "--covariate", "sex", "--measure", "density")
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "opt" / "outcome" / "outcome.csv").exists()
    assert _run("pca", "-i", opt, "--measures", "density", "global_efficiency(binary)", "global_efficiency(weighted)",
                "modularity", "--variables", str(tmp_path / "participants.tsv"), "--predictor", "age").returncode == 0
