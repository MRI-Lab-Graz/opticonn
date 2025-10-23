"""Runtime utilities for OptiConn CLI scripts.

Provides helpers to keep console output resilient across platforms.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def configure_stdio(no_emoji: Optional[bool] = None) -> bool:
    """Configure stdout/stderr to tolerate wide characters.

    Parameters
    ----------
    no_emoji:
        Deprecated parameter for backward compatibility. No longer used.

    Returns
    -------
    bool
        Always returns False (no_emoji is no longer supported).
    """

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

    return False


def no_emoji_enabled() -> bool:
    """Return whether console emoji suppression is active. Always returns False."""
    return False


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
    """Return an environment dict carrying Qt offscreen mode and DSI Studio path."""
    env = dict(os.environ if env is None else env)
    # Enable Qt offscreen mode for DSI Studio on headless servers
    env["QT_QPA_PLATFORM"] = "offscreen"
    # Propagate DSI_STUDIO_PATH if it's set in the environment
    if "DSI_STUDIO_PATH" in os.environ:
        env["DSI_STUDIO_PATH"] = os.environ["DSI_STUDIO_PATH"]
    return env
