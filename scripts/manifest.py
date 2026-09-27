"""One shared, tested contract for a stage boundary: the producer writes
what it made and any facts the consumer must check; the consumer verifies
before trusting it. See docs/superpowers/specs/2026-09-27-stage-manifest-contracts-design.md.

    from scripts.manifest import write_manifest, verify_manifest
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(dir: Path, stage: str, outputs: dict[str, Path], contract: dict) -> Path:
    """Hash every path in `outputs`, write dir/manifest.json, return its path."""
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    data = {
        "stage": stage,
        "produced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "outputs": {name: _sha256(Path(path)) for name, path in outputs.items()},
        "contract": contract,
    }
    path = dir / "manifest.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def verify_manifest(dir: Path, expected: dict, filename: str = "manifest.json") -> list[str]:
    """Check dir/filename against `expected`. [] means OK; never raises.

    Every output listed under "outputs" is re-hashed and must exist and match.
    Every key of `expected` is checked against the manifest's "contract" dict
    (if present) merged with every other top-level key -- so a flat JSON file
    with no "contract"/"outputs" nesting (e.g. OptiConn's own preflight.json)
    is checked directly, with no reshaping needed.
    """
    path = Path(dir) / filename
    if not path.exists():
        return [f"{filename} is missing in {dir}"]
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return [f"{filename} in {dir} is not valid JSON"]

    problems = []
    for name, want in (data.get("outputs") or {}).items():
        candidate = Path(dir) / name
        if not candidate.exists():
            problems.append(f"{name} is missing")
        elif _sha256(candidate) != want:
            problems.append(f"{name} does not match its manifest hash")

    facts = dict(data.get("contract") or {})
    for key, value in data.items():
        if key not in ("contract", "outputs"):
            facts[key] = value
    for key, want in expected.items():
        got = facts.get(key, "<missing>")
        if got != want:
            problems.append(f"{key} is '{got}', expected '{want}'")
    return problems
