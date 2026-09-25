import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from scripts.ordering_viewer import ReferenceMissing, collect

N = 16


def _matrix(rng, scale):
    """A symmetric matrix with exactly `scale` edges (out of N*(N-1)/2 possible),
    so density is a deterministic function of `scale` at any seed -- subjects
    ordered by scale are ordered the same way on density, with no ties, and
    density carries no tracking noise by construction. Which specific edges are
    chosen, and their weights, are still randomised by `rng`, so topology- and
    weight-dependent measures (e.g. global_efficiency, clustering) still carry
    real noise across repeats and differ across independently-seeded combos."""
    idx_i, idx_j = np.triu_indices(N, k=1)
    n_edges = min(int(scale), len(idx_i))
    chosen = rng.permutation(len(idx_i))[:n_edges]
    weights = rng.random(n_edges) * scale
    m = np.zeros((N, N))
    m[idx_i[chosen], idx_j[chosen]] = weights
    m[idx_j[chosen], idx_i[chosen]] = weights
    return m


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


def _sweep(tmp_path, seed=0):
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


def test_rho_noise_and_reference_differ_for_a_moved_candidate(tmp_path):
    """sweep_0002 is built from a different seed than the reference, so on a
    topology-dependent measure (unlike density, or global_efficiency here, whose
    values track edge *count* -- a fixed function of scale in this fixture --
    almost perfectly) its tracking-noise floor and its displacement from the
    reference are two genuinely different numbers -- catching a regression that
    self-compares (`_rho(rep1, rep1)`) or points `ref_id` at `cid`."""
    payload = collect(_sweep(tmp_path))
    combo = next(c for c in payload["combos"] if c["id"] == "wave1/sweep_0002")
    measure = "clustering_coeff_average(binary)"
    rho_noise = combo["rho_noise"]["AAL3/count"][measure]
    rho_ref = combo["rho_vs_reference"]["AAL3/count"][measure]
    assert rho_noise is not None and np.isfinite(rho_noise)
    assert rho_ref is not None and np.isfinite(rho_ref)
    assert rho_ref != rho_noise


def test_ties_produce_averaged_ranks(tmp_path):
    """Two subjects sharing the same scale get the same edge count -- an exact
    tie in density -- and the payload must emit rankdata's averaged rank for
    both, not an arbitrary tie-break, since the schema promises floats for
    exactly this case."""
    scales = {"sub-01": 10.0, "sub-02": 20.0, "sub-03": 20.0}
    _write_combo(tmp_path, "wave1", 1, {"fa_threshold": 0.05}, False, scales, 0)
    _write_combo(tmp_path, "wave1", 2, {"fa_threshold": 0.0}, True, scales, 0)
    payload = collect(tmp_path)
    combo = next(c for c in payload["combos"] if c["id"] == "wave1/sweep_0001")
    ranks = combo["ranks"]["AAL3/count"]["density"]
    assert sorted(ranks) == [1.0, 2.5, 2.5]


def test_wave_scoped_reference_does_not_cross_waves(tmp_path):
    """A second wave with its own, disjoint subject set and its own reference
    must resolve independently -- wave2's combos must never be compared
    against wave1's reference, and vice versa."""
    root = _sweep(tmp_path)
    scales2 = {f"w2-sub-{i:02d}": 10.0 * i for i in range(1, 5)}
    _write_combo(root, "wave2", 1, {"fa_threshold": 0.2}, False, scales2, seed=50)
    _write_combo(root, "wave2", 2, {"fa_threshold": 0.0}, True, scales2, seed=50)
    payload = collect(root)
    assert payload["waves"] == ["wave1", "wave2"]
    assert payload["reference"]["wave2"] == "wave2/sweep_0002"
    assert payload["reference"]["wave1"] != payload["reference"]["wave2"]
    assert len(payload["subjects"]["wave2"]) == 4
    wave2_combo = next(c for c in payload["combos"] if c["id"] == "wave2/sweep_0001")
    assert len(wave2_combo["ranks"]["AAL3/count"]["density"]) == 4
