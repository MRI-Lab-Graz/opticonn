"""Regression tests for the Windows UnicodeEncodeError ('charmap' codec) bug.

On Windows, `logging.FileHandler()` without an explicit encoding falls back to
the system locale codec (cp1252 / "charmap"), which cannot encode emoji or
other non-Latin-1 characters. `sys.stdout`/`sys.stderr` have the same problem
for console output. See MRI-Lab-Graz/opticonn issue #2.
"""

import ast
import io
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories whose Python files ship to users and must be Windows-safe.
SHIPPED_DIRS = ["scripts", "dsi_studio_tools", "opticonn.py"]

EMOJI_PATTERN = re.compile(
    r"[\U0001F000-\U0001FFFF☀-➿⬀-⯿←-⇿✅❌]"
)


def _shipped_py_files():
    for entry in SHIPPED_DIRS:
        path = REPO_ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")


def test_no_filehandler_without_utf8_encoding():
    """Every logging.FileHandler(...) call must pass encoding="utf-8"."""
    offenders = []
    for path in _shipped_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_filehandler = (isinstance(func, ast.Attribute) and func.attr == "FileHandler") or (
                isinstance(func, ast.Name) and func.id == "FileHandler"
            )
            if not is_filehandler:
                continue
            has_encoding = any(kw.arg == "encoding" for kw in node.keywords)
            if not has_encoding:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, (
        "FileHandler() calls missing encoding=\"utf-8\" (will crash on Windows "
        f"with UnicodeEncodeError when logging non-ASCII text): {offenders}"
    )


def test_no_decorative_emoji_in_shipped_scripts():
    """Shipped scripts must not contain emoji/arrow characters that crash
    under the Windows default 'charmap' codec."""
    offenders = []
    for path in _shipped_py_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if EMOJI_PATTERN.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()!r}")

    assert not offenders, f"Non-ASCII decorative characters found: {offenders}"


def test_configure_stdio_prevents_unicode_crash_on_narrow_encoding():
    """configure_stdio() must make stdout tolerate characters unencodable in
    the stream's current codec instead of raising UnicodeEncodeError."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.utils.runtime import configure_stdio

    buffer = io.BytesIO()
    narrow_stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")

    original_stdout = sys.stdout
    sys.stdout = narrow_stream
    try:
        configure_stdio()
        # Emoji is not representable in cp1252; without the fix this raises
        # UnicodeEncodeError (the exact bug reported in issue #2).
        print("Log file ready \U0001F4C4 ❌ →", file=sys.stdout)
        sys.stdout.flush()
    finally:
        sys.stdout = original_stdout
