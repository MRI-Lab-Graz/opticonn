"""Shared file discovery for the DSI Studio and Bayesian optimizer backends.

Both backends used to run their own `rglob` calls directly, which drifted:
one gained an `is_file()` guard the other didn't, neither excluded `.git`.
Under a DataLad/git-annex dataset, `rglob("*.fz")` returns both the friendly
symlink (`sub-079/fib/sub-079_ses-1.odf.qsdr.fz`) and the annex object it
resolves to (`.git/annex/objects/xx/yy/KEY.fz`, itself a directory containing
a same-named file one level deeper) -- both satisfy `is_file()`, so every
scan was discovered twice under two different, non-obviously-related names.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path


def find_subject_files(root: Path, patterns: list[str]) -> list[Path]:
    """Union of `root.rglob(pattern)` for each pattern, excluding VCS internals
    and de-duplicated by resolved path (first occurrence, in pattern order,
    then sorted order, wins and is what's returned -- not the resolved path).
    """
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if ".git" in path.parts:
                continue
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return out


_SUBJECT_RE = re.compile(r"sub-([A-Za-z0-9]+)")
_SESSION_RE = re.compile(r"ses-([A-Za-z0-9]+)")


def parse_subject_session(path: str | Path) -> tuple[str | None, str | None]:
    """Parse `sub-<id>` and `ses-<id>` from the string form of `path`.

    Prefers identifiers from the filename; falls back to the full path if not
    found in the filename.

    Returns (None, None) for names with no `sub-` match at all, such as a
    git-annex content hash (`MD5E-s6751...`).
    """
    path_obj = Path(path)
    filename = path_obj.name
    full_text = str(path)

    # Search filename first
    subject_match = _SUBJECT_RE.search(filename)
    session_match = _SESSION_RE.search(filename)

    # Fall back to full path for whichever isn't found in filename
    if not subject_match:
        subject_match = _SUBJECT_RE.search(full_text)
    if not session_match:
        session_match = _SESSION_RE.search(full_text)

    subject = f"sub-{subject_match.group(1)}" if subject_match else None
    session = f"ses-{session_match.group(1)}" if session_match else None
    return subject, session


def _natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def select_scans(
    pool: list[Path], n_subjects: int, seed: int, sessions_per_subject: int = 0
) -> list[Path]:
    """Choose the scans a wave stages.

    sessions_per_subject < 2 reproduces the legacy behaviour exactly: sample
    n_subjects individual scans (the whole pool when asked for at least as many).
    With sessions_per_subject >= 2, sample n_subjects *subjects* that have at
    least that many sessions and stage their first sessions_per_subject sessions
    (natural order of the session id), so within-subject between-session
    comparisons exist. Falls back to scan-level sampling when no subject has
    enough sessions, so single-session cohorts behave exactly as before.
    """

    def scan_level() -> list[Path]:
        if n_subjects >= len(pool):
            return list(pool)
        return random.Random(seed).sample(pool, n_subjects)

    if sessions_per_subject < 2:
        return scan_level()

    by_subject: dict[str, dict[str, Path]] = {}
    unparseable: list[Path] = []
    for path in pool:
        subject, session = parse_subject_session(path)
        if not subject or not session:
            unparseable.append(path)
            continue
        by_subject.setdefault(subject, {}).setdefault(session, path)  # first (.fz) wins
    if unparseable:
        logging.warning(
            "%d of %d scans had no parseable sub-<id>/ses-<id> and cannot be used "
            "for session-aware selection (e.g. %s)",
            len(unparseable),
            len(pool),
            ", ".join(p.name for p in unparseable[:3]),
        )
    eligible = sorted(s for s, sessions in by_subject.items() if len(sessions) >= sessions_per_subject)
    if not eligible:
        logging.info(
            "No subject has >=%d sessions; falling back to sampling individual scans",
            sessions_per_subject,
        )
        return scan_level()
    chosen = (
        eligible
        if n_subjects >= len(eligible)
        else sorted(random.Random(seed).sample(eligible, n_subjects))
    )
    selected: list[Path] = []
    for subject in chosen:
        sessions = sorted(by_subject[subject], key=_natural_key)[:sessions_per_subject]
        selected.extend(by_subject[subject][s] for s in sessions)
    return selected
