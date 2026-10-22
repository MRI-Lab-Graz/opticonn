"""Read a per-subject variables table (a BIDS participants.tsv works) for outcome analyses."""

from __future__ import annotations

import csv
from pathlib import Path

MISSING = {"", "n/a", "na", "nan", "none"}


def subject_id(raw: str) -> str:
    """Normalise a participant id to the 'sub-<id>' form of scan names."""
    raw = str(raw).strip()
    return raw if raw.startswith("sub-") else f"sub-{raw}"


def _read_table(path) -> list[dict]:
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _value(raw):
    raw = (raw or "").strip()
    return None if raw.lower() in MISSING else raw


def load_variables(path, columns: list[str]) -> tuple[dict, dict]:
    """({subject: {column: float | None}}, {column: {"0": level, "1": level}}).

    Numeric columns are kept. A non-numeric column with exactly two distinct values is coded
    0/1 in sorted order (the coding is returned); any other non-numeric column is an error.
    """
    rows = _read_table(path)
    if not rows or "participant_id" not in rows[0]:
        raise ValueError(f"{path}: a 'participant_id' column is required")
    absent = [c for c in columns if c not in rows[0]]
    if absent:
        raise ValueError(f"{path}: no column named {', '.join(absent)}; columns are {', '.join(rows[0])}")
    table = {subject_id(r["participant_id"]): {c: _value(r[c]) for c in columns} for r in rows}
    coding = {}
    for c in dict.fromkeys(columns):
        raw = {sid: v[c] for sid, v in table.items()}
        try:
            coded = {sid: None if s is None else float(s) for sid, s in raw.items()}
        except ValueError:
            levels = sorted({s for s in raw.values() if s is not None})
            if len(levels) != 2:
                raise ValueError(f"column '{c}' has {len(levels)} levels ({', '.join(levels[:4])}): "
                                 "only numeric or two-level columns are supported")
            coding[c] = {"0": levels[0], "1": levels[1]}
            coded = {sid: None if s is None else float(s == levels[1]) for sid, s in raw.items()}
        for sid, x in coded.items():
            table[sid][c] = x
    return table, coding
