import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from scripts.ordering_viewer import ReferenceMissing, collect

N = 16


def _matrix(rng, scale):
    """A symmetric matrix whose overall weight is set by `scale`, so subjects
    ordered by scale are ordered the same way on density-like measures."""
    mask = np.triu(rng.random((N, N)) < 0.4, 1)
    upper = np.where(mask, rng.random((N, N)) * scale, 0.0)
    return upper + upper.T


def _write_combo(root, wave, index, params, reference, scales, seed):
    combo = root / wave / "combos" / f"sweep_{index:04d}"
    for rep, rep_name in enumerate(("rep_1", "rep_2"), start=1):
        for s_i, (subject, scale) in enumerate(sorted(scales.items())):
            # collect_matrices() (scripts/reliability.py) takes the atlas from the
            # immediate parent directory name, and the metric from the filename
            # segment before "..pass.connectivity.mat".
            d = combo / rep_name / "AAL3"
            d.mkdir(parents=True, exist_ok=True)
            # Deterministic across processes: str hashing is salted, s_i is not.
            rng = np.random.default_rng(seed + rep * 100 + s_i)
            scipy.io.savemat(
                d / f"{subject}.count..pass.connectivity.mat",
                {"connectivity": _matrix(rng, scale)},
            )
    (combo / "diagnostics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "wave": wave,
                "combo_index": index,
                "sampler": "grid",
                "reference": reference,
                "parameters": params,
            }
        )
    )
    return combo


def _sweep(tmp_path, seed=6):
    scales = {f"sub-{i:02d}": 10.0 * i for i in range(1, 6)}
    _write_combo(tmp_path, "wave1", 1, {"fa_threshold": 0.05}, False, scales, seed)
    _write_combo(tmp_path, "wave1", 2, {"fa_threshold": 0.10}, False, scales, seed + 1)
    _write_combo(tmp_path, "wave1", 3, {"fa_threshold": 0.0}, True, scales, seed)
    return tmp_path


def test_collect_anonymizes_subjects(tmp_path):
    payload = collect(_sweep(tmp_path))
    assert payload["subjects"]["wave1"] == ["S01", "S02", "S03", "S04", "S05"]
    assert "sub-01" not in json.dumps(payload)


def test_collect_identifies_the_reference_per_wave(tmp_path):
    payload = collect(_sweep(tmp_path))
    assert payload["reference"]["wave1"] == "wave1/sweep_0003"
    flagged = [c["id"] for c in payload["combos"] if c["reference"]]
    assert flagged == ["wave1/sweep_0003"]


def test_combo_identical_to_reference_correlates_perfectly(tmp_path):
    """sweep_0001 and the reference are built from the same seed, so their
    subject ordering is identical and rho must be 1."""
    payload = collect(_sweep(tmp_path))
    combo = next(c for c in payload["combos"] if c["id"] == "wave1/sweep_0001")
    assert combo["rho_vs_reference"]["AAL3/count"]["density"] == pytest.approx(1.0)


def test_ranks_have_one_entry_per_subject(tmp_path):
    payload = collect(_sweep(tmp_path))
    combo = payload["combos"][0]
    ranks = combo["ranks"]["AAL3/count"]["density"]
    assert sorted(ranks) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_order_params_lists_only_parameters_that_vary(tmp_path):
    payload = collect(_sweep(tmp_path))
    assert payload["order_params"] == ["fa_threshold"]


def test_missing_reference_raises_naming_the_wave(tmp_path):
    scales = {f"sub-{i:02d}": 10.0 * i for i in range(1, 6)}
    _write_combo(tmp_path, "wave1", 1, {"fa_threshold": 0.05}, False, scales, 0)
    with pytest.raises(ReferenceMissing, match="wave1"):
        collect(tmp_path)


def test_too_few_subjects_raises(tmp_path):
    scales = {f"sub-{i:02d}": 10.0 * i for i in range(1, 3)}
    _write_combo(tmp_path, "wave1", 1, {"fa_threshold": 0.05}, False, scales, 0)
    _write_combo(tmp_path, "wave1", 2, {"fa_threshold": 0.0}, True, scales, 0)
    with pytest.raises(ValueError, match="3"):
        collect(tmp_path)
