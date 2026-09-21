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


def baseline_scans(pool: list[Path]) -> list[Path]:
    """One scan per subject: the first session in natural order (ses-2 before ses-10).

    OptiConn is cross-sectional: a later session usually carries the effect a
    study measures (e.g. an intervention), so it is never used to choose
    parameters. A subject without a session id contributes its scan as is. Paths
    whose subject does not parse (git-annex hashes, synthetic test ids) are kept,
    each counted as its own subject. When two paths share a subject and session
    (a .fz and its .fib.gz copy), the first listed wins. Output keeps input order.
    """
    first: dict[str, tuple[list, Path]] = {}
    unparseable: list[Path] = []
    for path in pool:
        subject, session = parse_subject_session(path)
        if subject is None:
            unparseable.append(path)
            continue
        key = _natural_key(session or "")
        if subject not in first or key < first[subject][0]:
            first[subject] = (key, path)
    if unparseable:
        logging.warning(
            "%d of %d scans have no parseable sub-<id>; each is kept as its own subject (e.g. %s)",
            len(unparseable),
            len(pool),
            ", ".join(p.name for p in unparseable[:3]),
        )
    keep = {path for _, path in first.values()} | set(unparseable)
    return [p for p in pool if p in keep]


def select_scans(pool: list[Path], n_subjects: int, seed: int) -> list[Path]:
    """Seeded sample of n_subjects scans, or the whole pool when it has no more.

    Pass a pool from baseline_scans() so each sampled scan is a different subject.
    """
    if n_subjects >= len(pool):
        return list(pool)
    return random.Random(seed).sample(pool, n_subjects)
