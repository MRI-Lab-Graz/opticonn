# Vendored runtime utilities (adapted from upstream pipeline for local use)

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
    if not isinstance(text, str):
        return text
    return _EMOJI_RE.sub("", text)


def _safe_print(*args, **kwargs):
    processed = []
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


def _ensure_print_hook():
    global _PRINT_PATCHED
    if not _PRINT_PATCHED:
        builtins.print = _safe_print
        _PRINT_PATCHED = True


def _ensure_logging_filter():
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
    global _NO_EMOJI
    if no_emoji is None:
        env_val = os.environ.get("OPTICONN_NO_EMOJI")
        if env_val is not None:
            no_emoji = env_val.lower() in ("1", "true", "yes", "on")
        else:
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


def propagate_no_emoji(env: Optional[dict[str, str]] = None) -> dict[str, str]:
    # Prepare an environment dict for subprocesses that preserves the no-emoji flag
    env = dict(os.environ if env is None else env)
    env["OPTICONN_NO_EMOJI"] = "1" if _NO_EMOJI else env.get("OPTICONN_NO_EMOJI", "0")
    # NOTE: We intentionally do NOT inject the repository root into PYTHONPATH
    # here. Subprocesses should be invoked using `python -m scripts.<module>` so
    # that the `scripts` package is importable without environment hacks. This
    # enforces stricter, more reproducible behavior and fails loudly if vendored
    # modules are missing.
    return env


def no_emoji_enabled() -> bool:
    """Return whether emoji output is currently disabled."""
    return bool(_NO_EMOJI)


def prepare_path_for_subprocess(p: str | os.PathLike | None) -> str | None:
    """Normalize a path for passing to subprocesses (absolute string) or return None.

    This mirrors the simple behavior expected by the vendored pipeline code.
    """
    if p is None:
        return None
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)
