import pandas as pd

from scripts.qc_gate import flag_low_outliers, load_qc


def _cohort(n=20, bad=None):
    vals = {f"sub-{i:03d}_ses-1": 2.0 + 0.01 * (i % 5) for i in range(n)}
    vals.update(bad or {})
    return pd.Series(vals)


def test_flags_only_low_outliers():
    v = _cohort(bad={"sub-900_ses-1": 1.25, "sub-901_ses-1": 2.6})
    assert list(flag_low_outliers(v).scan) == ["sub-900_ses-1"]


def test_small_cohort_is_not_flagged():
    assert flag_low_outliers(_cohort(n=5, bad={"sub-900_ses-1": 0.1})).empty


def test_load_qc_reads_tsv_and_skips_broken_symlinks(tmp_path):
    d = tmp_path / "sub-001" / "ses-1" / "dwi"
    d.mkdir(parents=True)
    (d / "sub-001_ses-1_acq-multi_space-ACPC_desc-image_qc.tsv").write_text(
        "t1post_dwi_contrast\n1.9\n"
    )
    (d / "sub-002_ses-1_desc-image_qc.tsv").symlink_to(tmp_path / "missing")
    assert load_qc(tmp_path).to_dict() == {"sub-001_ses-1": 1.9}
