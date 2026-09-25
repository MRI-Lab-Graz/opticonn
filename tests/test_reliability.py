import numpy as np
import pandas as pd
import pytest
import scipy.io

from scripts.reliability import (
    collect_matrices,
    discriminability,
    discriminability_margin,
    gate_reason,
    graph_stats,
    loo_top1_frequency,
    rank,
    repeatability,
    score_combo,
)

N_NODES = 20


def _sym(rng, scale=100.0):
    upper = np.triu(rng.random((N_NODES, N_NODES)) * scale, 1)
    return upper + upper.T


def _dataset(subject_specific, subjects=4, repeats=2, seed=0):
    """Repeats differ by small noise; subjects differ only if subject_specific."""
    rng = np.random.default_rng(seed)
    shared = _sym(rng)
    data = {}
    for s in range(subjects):
        truth = _sym(rng) if subject_specific else shared
        data[f"sub{s}"] = [truth + _sym(rng, scale=5.0) for _ in range(repeats)]
    return data


def test_subject_specific_connectomes_are_discriminable():
    assert discriminability(_dataset(subject_specific=True)) > 0.95


def test_connectomes_without_subject_signal_score_near_chance():
    # The old score gave near-identical values to very different settings;
    # this is the check that the new one does not.
    assert discriminability(_dataset(subject_specific=False, subjects=10, repeats=3)) < 0.7


def test_identical_connectomes_are_a_tie_not_a_win():
    m = _sym(np.random.default_rng(1))
    assert discriminability({"a": [m, m], "b": [m, m]}) == 0.5


def test_discriminability_undefined_without_two_subjects_with_repeats():
    m = _sym(np.random.default_rng(1))
    assert np.isnan(discriminability({"a": [m, m]}))
    assert np.isnan(discriminability({"a": [m], "b": [m]}))


def test_repeatability_is_one_for_identical_repeats():
    m = _sym(np.random.default_rng(2))
    assert repeatability({"a": [m, m]}) == pytest.approx(1.0)


def test_graph_stats_density_and_isolated_nodes():
    m = np.ones((4, 4))
    np.fill_diagonal(m, 0)
    m[3, :] = 0
    m[:, 3] = 0
    density, isolated = graph_stats({"a": [m]})
    assert density == 0.5  # 3 of 6 possible edges
    assert isolated == 0.25  # node 3


def test_gate_reason():
    assert gate_reason(0.3, 0.0, (0.02, 0.6), 0.1) == ""
    assert "density" in gate_reason(0.9, 0.0, (0.02, 0.6), 0.1)
    assert "isolated" in gate_reason(0.3, 0.5, (0.02, 0.6), 0.1)


def test_rank_drops_rejected_and_nan_then_orders():
    rows = [
        {"id": "low", "discriminability": 0.7, "repeatability": 0.99, "rejected": "", "tract_count": 1},
        {"id": "gated", "discriminability": 1.0, "repeatability": 0.99, "rejected": "density", "tract_count": 1},
        {"id": "tie_worse_rep", "discriminability": 0.9, "repeatability": 0.80, "rejected": "", "tract_count": 1},
        {"id": "best", "discriminability": 0.9, "repeatability": 0.95, "rejected": "", "tract_count": 1},
        {"id": "nan", "discriminability": float("nan"), "repeatability": 0.9, "rejected": "", "tract_count": 1},
    ]
    assert [r["id"] for r in rank(rows)] == ["best", "tie_worse_rep", "low"]


def test_loo_top1_frequency_prefers_the_discriminable_candidate():
    freq = loo_top1_frequency({"signal": _dataset(True), "noise": _dataset(False)})
    assert freq == {"signal": 1.0, "noise": 0.0}


def test_loo_top1_frequency_undefined_below_three_subjects():
    freq = loo_top1_frequency({"x": _dataset(True, subjects=2)})
    assert np.isnan(freq["x"])


def test_loo_top1_frequency_splits_exact_ties():
    data = _dataset(True)
    freq = loo_top1_frequency({"a": data, "b": data})
    assert freq == {"a": 0.5, "b": 0.5}


def test_loo_top1_frequency_breaks_discriminability_tie_by_repeatability():
    data = _dataset(True)
    # Same matrices (so discriminability ties exactly) but "b" gets an extra,
    # even-more-consistent repeat per subject so its repeatability is higher.
    tied_disc = {s: list(ms) for s, ms in data.items()}
    higher_repeatability = {s: list(ms) + [ms[0]] for s, ms in data.items()}
    freq = loo_top1_frequency(
        {"a": tied_disc, "b": higher_repeatability},
        tract_counts={"a": 1, "b": 1},
    )
    assert freq["b"] == 1.0
    assert freq["a"] == 0.0


def _write_dsi_mat(root, rep, subject, atlas, metric, matrix):
    d = root / f"rep_{rep}" / "01_connectivity" / f"{subject}.gqi_20250101" / "tracks_100k" / "results" / atlas
    d.mkdir(parents=True, exist_ok=True)
    name = f"{subject}.gqi_{atlas}.tt.gz.{atlas}.{metric}..pass.connectivity.mat"
    scipy.io.savemat(str(d / name), {"connectivity": matrix})


def test_collect_and_score_dsi_studio_layout(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        for k, m in enumerate(reps, 1):
            _write_dsi_mat(tmp_path, k, subject, "FreeSurferDKT_Cortical", "count", m)
    got = collect_matrices(tmp_path)
    assert list(got) == [("FreeSurferDKT_Cortical", "count")]
    assert sorted(got[("FreeSurferDKT_Cortical", "count")]) == ["sub0", "sub1", "sub2"]
    [row] = score_combo(tmp_path, {})
    assert row["n_scans"] == 3 and row["n_repeats"] == 2
    assert row["discriminability_margin"] > 0
    assert row["discriminability"] > 0.95


def _write_combined_dsi_mat(root, rep, subject, atlas, count_matrix, fa_matrix):
    """New DSI Studio layout: one combined <atlas>.connectivity.mat per subject/rep, no metric segment."""
    d = root / f"rep_{rep}" / "01_connectivity" / f"{subject}.gqi_20250101" / "tracks_100k" / "results" / atlas
    d.mkdir(parents=True, exist_ok=True)
    name = f"{subject}.gqi_{atlas}.tt.gz.{atlas}.connectivity.mat"
    scipy.io.savemat(
        str(d / name),
        {
            "number of tracts r2r": count_matrix,
            "dti_fa r2r": fa_matrix,
            "number of tracts t2r": count_matrix[:, :1],
        },
    )


def test_score_combo_rejects_subject_with_fewer_than_two_repeats(tmp_path):
    data = _dataset(True, subjects=3)
    for subject, reps in data.items():
        n_reps = 1 if subject == "sub0" else 2
        for k, m in enumerate(reps[:n_reps], 1):
            _write_dsi_mat(tmp_path, k, subject, "FreeSurferDKT_Cortical", "count", m)
    [row] = score_combo(tmp_path, {"density_range": [0.02, 1.0]})
    assert "fewer than 2 repeats" in row["rejected"]


def _write_mrtrix_csv(root, rep, subject, atlas, metric, matrix):
    """MRtrix3 backend layout: `scripts.utils.mrtrix.write_opticonn_connectivity_csv`
    output under <theta_dir>/rep_<k>/results/<atlas>/<subject>_<atlas>.<metric>.connectivity.csv
    (labelled rows/cols, not a .mat file)."""
    d = root / f"rep_{rep}" / "results" / atlas
    d.mkdir(parents=True, exist_ok=True)
    labels = [str(i) for i in range(matrix.shape[0])]
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(d / f"{subject}_{atlas}.{metric}.connectivity.csv")


def test_collect_and_score_mrtrix_csv_layout(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        for k, m in enumerate(reps, 1):
            _write_mrtrix_csv(tmp_path, k, subject, "Schaefer200", "count", m)
    got = collect_matrices(tmp_path)
    assert list(got) == [("Schaefer200", "count")]
    assert sorted(got[("Schaefer200", "count")]) == ["sub0", "sub1", "sub2"]
    [row] = score_combo(tmp_path, {})
    assert row["n_scans"] == 3 and row["n_repeats"] == 2
    assert row["discriminability"] > 0.95


def test_score_combo_rejects_unequal_repeats_across_subjects(tmp_path):
    data = _dataset(True, subjects=3, repeats=3)
    for subject, reps in data.items():
        n_reps = 2 if subject == "sub0" else 3
        for k, m in enumerate(reps[:n_reps], 1):
            _write_dsi_mat(tmp_path, k, subject, "FreeSurferDKT_Cortical", "count", m)
    [row] = score_combo(tmp_path, {"density_range": [0.02, 1.0]})
    assert "unequal repeats" in row["rejected"]


def test_collect_combined_dsi_studio_layout(tmp_path):
    dataset = _dataset(True, subjects=3)
    for subject, reps in dataset.items():
        for k, m in enumerate(reps, 1):
            _write_combined_dsi_mat(tmp_path, k, subject, "FreeSurferDKT_Cortical", m, m + 1.0)
    got = collect_matrices(tmp_path)
    assert set(got) == {("FreeSurferDKT_Cortical", "count"), ("FreeSurferDKT_Cortical", "fa")}
    assert sorted(got[("FreeSurferDKT_Cortical", "count")]) == ["sub0", "sub1", "sub2"]
    assert sorted(got[("FreeSurferDKT_Cortical", "fa")]) == ["sub0", "sub1", "sub2"]
    assert len(got[("FreeSurferDKT_Cortical", "count")]["sub0"]) == 2
    assert len(got[("FreeSurferDKT_Cortical", "fa")]["sub0"]) == 2
    # qa key absent from the fixture -> no qa entry at all
    assert ("FreeSurferDKT_Cortical", "qa") not in got


def test_resolve_repeats_warns_when_below_two(caplog):
    from scripts.reliability import resolve_repeats

    with caplog.at_level("WARNING"):
        assert resolve_repeats({"repeats": 1}) == 1
    assert "2 repeats" in caplog.text


def test_resolve_repeats_is_quiet_for_two_or_more(caplog):
    from scripts.reliability import resolve_repeats

    with caplog.at_level("WARNING"):
        assert resolve_repeats({"repeats": 3}) == 3
    assert caplog.text == ""


def test_collect_dedupes_combined_mat_and_per_metric_csv(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        for k, m in enumerate(reps, 1):
            _write_combined_dsi_mat(tmp_path, k, subject, "AAL3", m, m + 1.0)
            # DSI Studio also converts the count matrix to a per-metric CSV
            _write_mrtrix_csv(tmp_path, k, subject, "AAL3", "count", m)
    got = collect_matrices(tmp_path)
    assert len(got[("AAL3", "count")]["sub0"]) == 2  # was 4: .mat + .csv per repeat
    assert len(got[("AAL3", "fa")]["sub0"]) == 2
    [count_row] = [r for r in score_combo(tmp_path, {}) if r["connectivity_metric"] == "count"]
    assert count_row["n_repeats"] == 2


def test_collect_csv_still_supplies_metrics_the_mat_lacks(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        for k, m in enumerate(reps, 1):
            _write_combined_dsi_mat(tmp_path, k, subject, "AAL3", m, m + 1.0)
            _write_mrtrix_csv(tmp_path, k, subject, "AAL3", "ncount2", m)  # not a combined-mat metric
    got = collect_matrices(tmp_path)
    assert len(got[("AAL3", "ncount2")]["sub0"]) == 2
    assert len(got[("AAL3", "count")]["sub0"]) == 2


def test_single_surviving_repeat_is_not_padded_to_two_by_its_csv_copy(tmp_path):
    for subject, reps in _dataset(True, subjects=3).items():
        n_reps = 1 if subject == "sub0" else 2
        for k, m in enumerate(reps[:n_reps], 1):
            _write_combined_dsi_mat(tmp_path, k, subject, "AAL3", m, m + 1.0)
            _write_mrtrix_csv(tmp_path, k, subject, "AAL3", "count", m)
    [count_row] = [
        r for r in score_combo(tmp_path, {"density_range": [0.02, 1.0]})
        if r["connectivity_metric"] == "count"
    ]
    assert "fewer than 2 repeats" in count_row["rejected"]


def test_same_scan_written_twice_in_one_repeat_counts_once(tmp_path):
    rng_data = _dataset(True, subjects=3)
    for subject, reps in rng_data.items():
        for k, m in enumerate(reps, 1):
            _write_dsi_mat(tmp_path, k, subject, "AAL3", "count", m)
    # a second timestamped output dir for the same scan in repeat 1
    extra = tmp_path / "rep_1" / "01_connectivity" / "sub0.gqi_20250102" / "tracks_100k" / "results" / "AAL3"
    extra.mkdir(parents=True)
    scipy.io.savemat(
        str(extra / "sub0.gqi_AAL3.tt.gz.AAL3.count..pass.connectivity.mat"),
        {"connectivity": rng_data["sub0"][0]},
    )
    got = collect_matrices(tmp_path)
    assert len(got[("AAL3", "count")]["sub0"]) == 2


def test_newest_timestamped_dir_wins_a_same_scan_tie(tmp_path):
    old, new = np.zeros((4, 4)), np.ones((4, 4))
    for stamp, m in (("20250101", old), ("20250102", new)):
        d = tmp_path / "rep_1" / "01_connectivity" / f"sub0.gqi_{stamp}" / "tracks_100k" / "results" / "AAL3"
        d.mkdir(parents=True)
        scipy.io.savemat(str(d / "sub0.gqi_AAL3.tt.gz.AAL3.count..pass.connectivity.mat"), {"connectivity": m})
    got = collect_matrices(tmp_path)[("AAL3", "count")]["sub0"]
    assert len(got) == 1 and np.array_equal(got[0], new)


def test_margin_grows_with_subject_separation_when_discriminability_is_saturated():
    close = _dataset(subject_specific=True, seed=3)
    rng = np.random.default_rng(3)
    shared = _sym(rng)
    # same subjects but pulled toward a shared connectome: still identifiable, smaller margin
    near = {s: [0.7 * shared + 0.3 * m for m in ms] for s, ms in close.items()}
    assert discriminability(close) == discriminability(near) == 1.0
    assert discriminability_margin(close) > discriminability_margin(near) > 0


def test_margin_undefined_without_two_subjects_with_repeats():
    assert np.isnan(discriminability_margin(_dataset(True, subjects=1)))


def test_rank_orders_by_margin_when_discriminability_ties_and_puts_nan_margin_last():
    def row(i, margin, rep):
        return {"id": i, "discriminability": 1.0, "discriminability_margin": margin,
                "repeatability": rep, "rejected": "", "tract_count": 1}

    rows = [row("nan", float("nan"), 0.999), row("small", 0.1, 0.999), row("big", 0.3, 0.95)]
    assert [r["id"] for r in rank(rows)] == ["big", "small", "nan"]


def test_rank_excludes_reference_rows():
    from scripts.reliability import rank, rank_with_fallback

    candidate = {
        "rejected": "", "discriminability": 0.8, "discriminability_margin": 0.1,
        "repeatability": 0.5, "tract_count": 5000,
    }
    reference = {
        "rejected": "", "discriminability": 1.0, "discriminability_margin": 0.9,
        "repeatability": 0.9, "tract_count": 5000, "reference": True,
    }

    ranked = rank([reference, candidate])
    assert [r["discriminability"] for r in ranked] == [0.8]
    assert rank_with_fallback([reference, candidate])["discriminability"] == 0.8


def test_rank_with_fallback_excludes_reference_when_discriminability_is_nan():
    from scripts.reliability import rank_with_fallback

    candidate = {
        "rejected": "", "discriminability": float("nan"),
        "repeatability": 0.4, "quality_score_raw": 0.1,
    }
    reference = {
        "rejected": "", "discriminability": float("nan"),
        "repeatability": 0.99, "quality_score_raw": 0.9, "reference": True,
    }

    assert rank_with_fallback([reference, candidate])["repeatability"] == 0.4


def test_rank_with_fallback_returns_none_when_only_reference_is_usable():
    from scripts.reliability import rank_with_fallback

    reference = {
        "rejected": "", "discriminability": 1.0, "discriminability_margin": 0.5,
        "repeatability": 0.9, "tract_count": 5000, "reference": True,
    }
    assert rank_with_fallback([reference]) is None
