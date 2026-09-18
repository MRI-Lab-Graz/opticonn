from pathlib import Path

import pytest

from scripts.utils.discovery import find_subject_files, parse_subject_session


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
