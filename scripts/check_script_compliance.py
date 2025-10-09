#!/usr/bin/env python3
"""Repository script compliance checker

Checks scripts/ for adherence to the new global instructions:
 - each script should have a top-level docstring (README/usage)
 - each script should support a --dry-run flag
 - running without args should print help (parser.print_help or explicit check)

Usage:
  python scripts/check_script_compliance.py

Exit code: 0 = all compliant, 1 = non-compliant scripts found
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PY_FILES = sorted(
    [
        p
        for p in SCRIPTS_DIR.glob("*.py")
        if p.name != Path(__file__).name and not p.name.startswith("test_")
    ]
)

DOCSTRING_RE = re.compile(r'^\s*(?:"""|\'\'\')', re.M)
DRYRUN_RE = re.compile(r"--dry-run")
HELP_PRINT_RE = re.compile(
    r"parser\.print_help\(|len\(sys\.argv\)\s*==\s*1|parser\.print_help\(\)"
)

failures = []
reports = []

for p in PY_FILES:
    text = p.read_text(encoding="utf-8")
    has_doc = False
    has_dryrun = bool(DRYRUN_RE.search(text))
    prints_help = bool(HELP_PRINT_RE.search(text))

    # Detect module-level docstring: first non-empty token should be triple-quoted string
    stripped = text.lstrip()
    if stripped.startswith(('"""', "'''")):
        has_doc = True

    # Also accept README file alongside script: scripts/<scriptname>.md
    readme_path = p.with_suffix(".md")
    readme_exists = readme_path.exists()

    report = {
        "script": str(p.relative_to(SCRIPTS_DIR.parent)),
        "has_docstring": has_doc,
        "has_readme": readme_exists,
        "has_dry_run": has_dryrun,
        "prints_help_on_no_args": prints_help,
    }
    reports.append(report)

    # Evaluate compliance rules (conservative): either docstring OR separate README
    if not (has_doc or readme_exists or prints_help):
        failures.append((p.name, "Missing top-level docstring/README/help text"))
    if not has_dryrun:
        failures.append((p.name, "Missing --dry-run flag"))
    if not prints_help:
        # only warn if there's no explicit help-on-empty invocation; it's a recommendation
        failures.append(
            (
                p.name,
                "Does not print help when run without args (parser.print_help or explicit check)",
            )
        )

# Print human friendly table
print("\nScript compliance summary:\n")
fmt = "{:60s}  {:6s}  {:6s}  {:6s}  {:6s}"
print(fmt.format("script", "DOC", "MD", "DRYRN", "HELP"))
print("-" * 90)
for r in reports:
    print(
        fmt.format(
            r["script"],
            "yes" if r["has_docstring"] else "no",
            "yes" if r["has_readme"] else "no",
            "yes" if r["has_dry_run"] else "no",
            "yes" if r["prints_help_on_no_args"] else "no",
        )
    )

if failures:
    print("\nNon-compliant script warnings (or failures):")
    for name, reason in failures:
        print(f" - {name}: {reason}")
    print("\nNext steps:")
    print(
        " - Add a short top-level docstring or scripts/<script>.md with usage examples"
    )
    print(" - Add a --dry-run flag that performs a safe, no-side-effect run")
    print(
        " - Ensure running the script without arguments prints help (parser.print_help())"
    )
    sys.exit(1)

print("\nAll checked scripts appear to meet the basic guidelines.")
sys.exit(0)
