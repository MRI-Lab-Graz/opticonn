import logging
import random
from pathlib import Path

import pytest

from scripts.utils.discovery import find_subject_files, parse_subject_session, select_scans


def test_find_subject_files_excludes_git_paths(tmp_path):
    real = tmp_path / "sub-001" / "fib"
    real.mkdir(parents=True)
    real_file = real / "sub-001_ses-1.odf.qsdr.fz"
    real_file.write_bytes(b"data")

    annex_dir = tmp_path / "sub-001" / ".git" / "annex" / "objects" / "xx" / "yy" / "KEY.fz"
    annex_dir.mkdir(parents=True)
    (annex_dir / "KEY.fz").write_bytes(b"data")

    found = find_subject_files(tmp_path, ["*.fz"])

    assert found == [real_file]


def test_find_subject_files_dedupes_symlink_and_target(tmp_path):
    target_dir = tmp_path / "store"
    target_dir.mkdir()
    target = target_dir / "content.fz"
    target.write_bytes(b"data")

    friendly = tmp_path / "sub-001_ses-1.fz"
    friendly.symlink_to(target)

    found = find_subject_files(tmp_path, ["*.fz"])

    assert len(found) == 1


def test_find_subject_files_unions_multiple_patterns(tmp_path):
    (tmp_path / "a.fz").write_bytes(b"1")
    (tmp_path / "b.fib.gz").write_bytes(b"1")

    found = find_subject_files(tmp_path, ["*.fz", "*.fib.gz"])

    assert {p.name for p in found} == {"a.fz", "b.fib.gz"}


def test_find_subject_files_empty_dir_returns_empty_list(tmp_path):
    assert find_subject_files(tmp_path, ["*.fz"]) == []


@pytest.mark.parametrize(
    "path,expected",
    [
        ("sub-001_ses-1.odf.qsdr", ("sub-001", "ses-1")),
        ("/data/x/sub-079/fib/sub-079_ses-2.odf.qsdr.fz", ("sub-079", "ses-2")),
        ("sub-042", ("sub-042", None)),
        ("MD5E-s67517482--3e7da19b35854439806ebb6d2abc409b", (None, None)),
    ],
)
def test_parse_subject_session(path, expected):
    assert parse_subject_session(path) == expected


def test_parse_subject_session_accepts_path_object(tmp_path):
    p = tmp_path / "sub-005_ses-3.fz"
    assert parse_subject_session(p) == ("sub-005", "ses-3")


def test_parse_subject_session_prefers_filename_over_directory(tmp_path):
    # If parent directory and filename have different subject IDs,
    # the filename's identity should win. This ensures scans are not
    # silently mis-assigned to the wrong subject by a stale directory name.
    p = tmp_path / "sub-001" / "fib" / "sub-002_ses-1.odf.qsdr.fz"
    assert parse_subject_session(p) == ("sub-002", "ses-1")


def _scan(sub, ses, ext="fz"):
    return Path(f"/d/sub-{sub}/fib/sub-{sub}_ses-{ses}.odf.qsdr.{ext}")


_POOL = (
    [_scan("A", 1), _scan("A", 2), _scan("A", 3)]
    + [_scan("B", 1), _scan("B", 2)]
    + [_scan("C", 1)]
    + [_scan("D", 1), _scan("D", 2)]
)


def test_select_scans_matches_legacy_sampling_when_not_session_aware():
    pool = [_scan(str(i), 1) for i in range(10)]
    random.seed(42)
    legacy = random.sample(pool, 3)
    assert select_scans(pool, 3, 42, 0) == legacy
    assert select_scans(pool, 3, 42, 1) == legacy


def test_select_scans_uses_whole_pool_when_asked_for_more_than_exists():
    pool = [_scan(str(i), 1) for i in range(4)]
    assert select_scans(pool, 10, 42, 0) == pool


def test_select_scans_session_aware_takes_k_sessions_from_eligible_subjects_only():
    got = select_scans(_POOL, 2, 42, 2)
    assert len(got) == 4
    subjects = {p.name.split("_")[0] for p in got}
    assert len(subjects) == 2 and "sub-C" not in subjects  # C has a single session
    for s in subjects:
        assert sorted(p.name for p in got if p.name.startswith(s)) == [
            f"{s}_ses-1.odf.qsdr.fz",
            f"{s}_ses-2.odf.qsdr.fz",
        ]


def test_select_scans_is_deterministic():
    assert select_scans(_POOL, 2, 7, 2) == select_scans(_POOL, 2, 7, 2)


def test_select_scans_uses_all_eligible_subjects_when_n_exceeds_them():
    got = select_scans(_POOL, 99, 1, 2)
    assert {p.name.split("_")[0] for p in got} == {"sub-A", "sub-B", "sub-D"}
    assert len(got) == 6


def test_select_scans_falls_back_to_scan_sampling_without_multisession_subjects(caplog):
    pool = [_scan(str(i), 1) for i in range(10)]
    with caplog.at_level(logging.INFO):
        got = select_scans(pool, 3, 42, 2)
    random.seed(42)
    assert got == random.sample(pool, 3)
    assert "falling back" in caplog.text


def test_select_scans_skips_unparseable_names_with_a_warning(caplog):
    pool = _POOL + [Path("/d/.git/MD5E-abc.qsdr.fz")]
    with caplog.at_level(logging.WARNING):
        got = select_scans(pool, 99, 1, 2)
    assert all("MD5E" not in p.name for p in got)
    assert "1 of 9 scans" in caplog.text


def test_select_scans_prefers_fz_over_fib_gz_copy_of_the_same_scan():
    pool = [_scan("A", 1), _scan("A", 2), _scan("A", 1, "fib.gz"), _scan("A", 2, "fib.gz")]
    assert select_scans(pool, 1, 1, 2) == [_scan("A", 1), _scan("A", 2)]


def test_select_scans_orders_sessions_naturally():
    pool = [_scan("A", 1), _scan("A", 10), _scan("A", 2)]
    assert select_scans(pool, 1, 1, 2) == [_scan("A", 1), _scan("A", 2)]
