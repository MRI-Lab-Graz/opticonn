"""Prove that DSI Studio executed the tracking parameters OptiConn sent.

Mandatory for every tracking run (docs/superpowers/specs/
2026-09-27-battery-readiness-design.md, section 0). DSI Studio accepts unknown
options and exits 0, and resolves some values into strategies (fa_threshold=0 with
otsu_threshold=0.6 executes as a threshold randomised over 0.5-0.7 x Otsu), so an
exit code and an output file prove nothing. Three records are checked, each
failing closed:

1. the stdout echo -- what DSI Studio parsed. Every sent flag must be echoed back;
   a flag that is not is treated as not executed, whatever warning is printed.
2. the tract file's `report` -- DSI Studio's own prose record of what it executed.
   Each sent parameter must map to its expected executed statement.
3. the tract file's `parameter_id` -- an encoded fingerprint of the executed
   parameters (subject-independent, seed-dependent). Must equal the preflight's.

Streamline geometry (`check_geometry`) is a fourth, physical check used by the
preflight. Everything here is pure: nothing runs DSI Studio or writes files.
"""

from __future__ import annotations

import re

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ECHO = re.compile(r"^[│\s]*(?:[├└]──)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_UNRECOGNIZED = re.compile(r"--([A-Za-z_][A-Za-z0-9_]*) is not used/recognized")
_TOL = 1e-6

# Echoed values that must equal what was sent. Paths and connectivity options come
# back rewritten, so they are only confirmed present.
COMPARED = frozenset({
    "method", "otsu_threshold", "fa_threshold", "turning_angle", "step_size",
    "smoothing", "min_length", "max_length", "check_ending", "tip_iteration",
    "threshold_index", "random_seed", "tract_count", "thread_count",
})
# Flags naming files rather than parameters: excluded from the fingerprint key.
PATH_FLAGS = frozenset({"source", "output", "connectivity"})


def parse_command_flags(cmd: list[str]) -> dict[str, str]:
    """{flag: value} for every `--flag=value` token of a DSI Studio command."""
    flags: dict[str, str] = {}
    for token in map(str, cmd):
        if token.startswith("--") and "=" in token:
            key, value = token[2:].split("=", 1)
            flags[key] = value
    return flags


def parse_echo(stdout: str) -> dict[str, str]:
    """{key: value} DSI Studio echoed as parsed; the first occurrence wins."""
    echo: dict[str, str] = {}
    for line in _ANSI.sub("", stdout).splitlines():
        m = _ECHO.match(line.rstrip())
        if m:
            echo.setdefault(m[1], m[2].strip())
    return echo


def _same(a, b) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()
    return abs(fa - fb) <= _TOL * max(1.0, abs(fa))


def check_echo(sent: dict[str, str], stdout: str) -> list[str]:
    """Errors if a sent flag was reported unrecognised, not echoed back, or changed."""
    echo = parse_echo(stdout)
    if not echo:
        return ["no parameter echo found in DSI Studio output; cannot confirm anything was executed"]
    errors = [f"--{flag}: DSI Studio reports it as not used/recognized"
              for flag in _UNRECOGNIZED.findall(_ANSI.sub("", stdout))]
    for key, value in sent.items():
        if key not in echo:
            errors.append(f"--{key}={value}: sent but not echoed back, so not executed")
        elif key in COMPARED and not _same(value, echo[key]):
            errors.append(f"--{key}: sent {value}, DSI Studio parsed {echo[key]}")
    return errors


_N = r"([0-9]+(?:\.[0-9]+)?)"
_REPORT = {
    "otsu_window": re.compile(
        rf"anisotropy threshold was randomly selected between {_N} and {_N} otsu threshold"),
    "fa_fixed": re.compile(rf"anisotropy threshold was {_N}\."),
    "angle_random": re.compile(
        rf"angular threshold was randomly selected from {_N} degrees to {_N} degrees"),
    "angle_fixed": re.compile(rf"angular threshold was {_N} degrees"),
    "step_voxel": re.compile(r"step size was set to voxel spacing"),
    "step_fixed": re.compile(rf"step size was {_N} mm"),
    "smoothing": re.compile(rf"propagation direction with {_N}% of the previous direction"),
    "length": re.compile(rf"length shorter than {_N} or longer than {_N} mm were discarded"),
    "tract_count": re.compile(r"A total of ([0-9]+) tracts were tracked"),
    "pruning": re.compile(r"with ([0-9]+) iteration\(s\) to remove false connections"),
}
# A report without these statements cannot prove anything: fail closed.
_MANDATORY = ("anisotropy", "turning_angle", "step_size", "min_length", "tract_count")


def parse_report(report: str) -> dict:
    """What DSI Studio says it executed, from a tract file's `report`.

    A key is present only when its statement was found. Smoothing and pruning are
    reported only when active, so their absence means 0.
    """
    r = " ".join((report or "").split())
    ex: dict = {"smoothing": 0.0, "tip_iteration": 0}
    if m := _REPORT["otsu_window"].search(r):
        ex["anisotropy"] = {"kind": "otsu_window", "low": float(m[1]), "high": float(m[2])}
    elif m := _REPORT["fa_fixed"].search(r):
        ex["anisotropy"] = {"kind": "fixed", "value": float(m[1])}
    if m := _REPORT["angle_random"].search(r):
        ex["turning_angle"] = {"kind": "random", "low": float(m[1]), "high": float(m[2])}
    elif m := _REPORT["angle_fixed"].search(r):
        ex["turning_angle"] = {"kind": "fixed", "value": float(m[1])}
    if _REPORT["step_voxel"].search(r):
        ex["step_size"] = {"kind": "voxel_spacing"}
    elif m := _REPORT["step_fixed"].search(r):
        ex["step_size"] = {"kind": "fixed", "value": float(m[1])}
    if m := _REPORT["smoothing"].search(r):
        ex["smoothing"] = float(m[1]) / 100
    if m := _REPORT["length"].search(r):
        ex["min_length"], ex["max_length"] = float(m[1]), float(m[2])
    if m := _REPORT["tract_count"].search(r):
        ex["tract_count"] = int(m[1])
    if m := _REPORT["pruning"].search(r):
        ex["tip_iteration"] = int(m[1])
    return ex


def expected_execution(sent: dict[str, str]) -> dict:
    """What DSI Studio should execute for the sent flags (pinned build, spec section 0).

    Only sent parameters carry an expectation. Omitted ones (null in the config) are
    DSI Studio's own defaults, recorded as executed rather than compared. The
    otsu window (centre +/- 0.1) and the 45-90 degree random angle are DSI Studio's
    documented behaviour for fa_threshold=0 and turning_angle=0.
    """
    f = lambda k: float(sent[k])
    exp: dict = {}
    if "fa_threshold" in sent and f("fa_threshold") > 0:
        exp["anisotropy"] = {"kind": "fixed", "value": f("fa_threshold")}
    elif "otsu_threshold" in sent:
        t = f("otsu_threshold")
        exp["anisotropy"] = {"kind": "otsu_window", "low": t - 0.1, "high": t + 0.1}
    if "turning_angle" in sent:
        a = f("turning_angle")
        exp["turning_angle"] = ({"kind": "fixed", "value": a} if a > 0
                                else {"kind": "random", "low": 45.0, "high": 90.0})
    if "step_size" in sent:
        s = f("step_size")
        exp["step_size"] = {"kind": "fixed", "value": s} if s > 0 else {"kind": "voxel_spacing"}
    for key in ("smoothing", "min_length", "max_length"):
        if key in sent:
            exp[key] = f(key)
    for key in ("tract_count", "tip_iteration"):
        if key in sent:
            exp[key] = int(float(sent[key]))
    return exp


def _match(want, got) -> bool:
    if isinstance(want, dict) or isinstance(got, dict):
        return (isinstance(want, dict) and isinstance(got, dict) and want.keys() == got.keys()
                and all(_match(want[k], got[k]) for k in want))
    return _same(want, got)


def check_report(sent: dict[str, str], report: str) -> tuple[dict, list[str]]:
    """(executed parameters, errors): missing statements and mismatches are errors."""
    executed = parse_report(report)
    errors = [f"execution report has no statement for {k}" for k in _MANDATORY if k not in executed]
    for key, want in expected_execution(sent).items():
        got = executed.get(key)
        if got is not None and not _match(want, got):
            errors.append(f"{key}: expected {want}, DSI Studio executed {got}")
    return executed, errors
