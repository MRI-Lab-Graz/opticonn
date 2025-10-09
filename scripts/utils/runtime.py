"""Runtime utilities for OptiConn CLI scripts.

Provides helpers to keep console output resilient across platforms and to
optionally strip emoji characters when the user requests plain ASCII output
(e.g., on Windows terminals that cannot render them).
"""

from __future__ import annotations

import builtins
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FFFF\u2600-\u27FF\u2460-\u24FF]")

_ORIGINAL_PRINT = builtins.print
_PRINT_PATCHED = False
_LOG_FILTER_ADDED = False
_NO_EMOJI = False


def remove_emoji(text: str) -> str:
    """Strip emoji and miscellaneous symbols from *text*."""
    if not isinstance(text, str):
        return text
    return _EMOJI_RE.sub("", text)


def _safe_print(*args, **kwargs):  # pragma: no cover - runtime side effect
    processed: list[str] = []
    for arg in args:
        text = str(arg)
        if _NO_EMOJI:
            text = remove_emoji(text)
        processed.append(text)
    try:
        _ORIGINAL_PRINT(*processed, **kwargs)
    except UnicodeEncodeError:
        fallback = [t.encode("ascii", "replace").decode("ascii") for t in processed]
        _ORIGINAL_PRINT(*fallback, **kwargs)


def _ensure_print_hook():  # pragma: no cover - runtime side effect
    global _PRINT_PATCHED
    if not _PRINT_PATCHED:
        builtins.print = _safe_print
        _PRINT_PATCHED = True


def _ensure_logging_filter():  # pragma: no cover - runtime side effect
    global _LOG_FILTER_ADDED
    if _LOG_FILTER_ADDED:
        return

    class EmojiFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if _NO_EMOJI:
                if isinstance(record.msg, str):
                    record.msg = remove_emoji(record.msg)
                if record.args:
                    record.args = tuple(
                        remove_emoji(arg) if isinstance(arg, str) else arg
                        for arg in record.args
                    )
            return True

    logging.getLogger().addFilter(EmojiFilter())
    _LOG_FILTER_ADDED = True


def configure_stdio(no_emoji: Optional[bool] = None) -> bool:
    """Configure stdout/stderr to tolerate wide characters and optional emoji stripping.

    Parameters
    ----------
    no_emoji:
        If *True*, remove emoji from console output and store OPTICONN_NO_EMOJI=1 in the
        environment so child processes inherit the preference. If *False*, disable the
        environment hint. If *None*, infer the preference from the environment.

    Returns
    -------
    bool
        Current no-emoji state after configuration.
    """

    global _NO_EMOJI

    if no_emoji is None:
        env_val = os.environ.get("OPTICONN_NO_EMOJI")
        if env_val is not None:
            no_emoji = env_val.lower() in ("1", "true", "yes", "on")
        else:
            # Windows console encodings frequently cannot render emoji; default to stripping
            no_emoji = os.name == "nt"
            if no_emoji:
                os.environ["OPTICONN_NO_EMOJI"] = "1"
    else:
        os.environ["OPTICONN_NO_EMOJI"] = "1" if no_emoji else "0"

    _NO_EMOJI = bool(no_emoji)
    os.environ["OPTICONN_NO_EMOJI"] = "1" if _NO_EMOJI else "0"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

    _ensure_print_hook()
    _ensure_logging_filter()
    return _NO_EMOJI


def no_emoji_enabled() -> bool:
    """Return whether console emoji suppression is active."""
    return _NO_EMOJI


def prepare_path_for_subprocess(path: str | os.PathLike[str]) -> str:
    """Return a platform-appropriate path string for subprocess arguments."""
    p = Path(path)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    path_str = str(resolved)

    if os.name != "nt":
        return path_str

    if len(path_str) < 240:
        return path_str

    # Try to obtain 8.3 short path name when available to keep things compatible
    try:
        import ctypes

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint

        buffer_len = 260
        while True:
            buffer = ctypes.create_unicode_buffer(buffer_len)
            needed = GetShortPathNameW(path_str, buffer, buffer_len)
            if needed == 0:
                break
            if needed < buffer_len:
                short_path = buffer.value
                if short_path:
                    return short_path
                break
            buffer_len = needed + 1
    except Exception:
        pass

    # Fall back to extended-length path prefix
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC" + path_str[1:]
    if path_str.startswith("\\\\?\\"):
        return path_str
    return "\\\\?\\" + path_str


def propagate_no_emoji(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Return an environment dict carrying the current no-emoji preference."""
    env = dict(os.environ if env is None else env)
    env["OPTICONN_NO_EMOJI"] = "1" if _NO_EMOJI else env.get("OPTICONN_NO_EMOJI", "0")
    return env
