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
