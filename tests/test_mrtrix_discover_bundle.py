"""Tests for scripts/mrtrix_discover_bundle.py QSIRecon/qsiprep auto-discovery."""

import pytest

from scripts.mrtrix_discover_bundle import build_config_json, discover_bundle

ATLAS = "AtlasX"


def _touch(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _add_dseg_and_labels(dwi_dir, subject, session, atlas=ATLAS):
    _touch(dwi_dir / f"{subject}_{session}_space-ACPC_seg-{atlas}_dseg.mif.gz")
    _touch(dwi_dir / f"{subject}_{session}_seg-{atlas}_dseg.txt")


def _add_act(anat_dir, subject, session):
    _touch(anat_dir / f"{subject}_{session}_space-ACPC_seg-hsvs_probseg.nii.gz")


def test_discovers_wm_fod_under_derivatives_workflow_dir(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"
    workflow = qsirecon_dir / "derivatives" / "qsirecon-MRtrix3_act-HSVS"
    dwi_dir = workflow / subject / session / "dwi"
    _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    # dseg/labels are always resolved directly under qsirecon_dir/subject, not the workflow dir.
    _add_dseg_and_labels(qsirecon_dir / subject / session / "dwi", subject, session)
    _add_act(qsirecon_dir / subject / "anat", subject, session)

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=session,
        atlas=ATLAS,
        workflow_hint=None,
        allow_missing_act=False,
    )

    assert found.wm_fod == dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz"


def test_discovers_wm_fod_fallback_without_derivatives_layer(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"
    dwi_dir = qsirecon_dir / subject / session / "dwi"
    _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif")
    _add_dseg_and_labels(dwi_dir, subject, session)
    _add_act(qsirecon_dir / subject / "anat", subject, session)

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=session,
        atlas=ATLAS,
        workflow_hint=None,
        allow_missing_act=False,
    )

    assert found.wm_fod == dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif"


def test_workflow_hint_disambiguates_multiple_workflow_dirs(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"

    wf_a = qsirecon_dir / "derivatives" / "qsirecon-MRtrix3_act-HSVS"
    wf_b = qsirecon_dir / "derivatives" / "qsirecon-DSIStudio"

    dwi_a = wf_a / subject / session / "dwi"
    dwi_b = wf_b / subject / session / "dwi"
    _touch(dwi_a / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    _touch(dwi_b / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    # dseg/labels are always resolved directly under qsirecon_dir/subject, not the workflow dir.
    _add_dseg_and_labels(qsirecon_dir / subject / session / "dwi", subject, session)
    _add_act(qsirecon_dir / subject / "anat", subject, session)

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=session,
        atlas=ATLAS,
        workflow_hint="MRtrix3",
        allow_missing_act=False,
    )

    assert found.wm_fod == dwi_a / f"{subject}_{session}_label-WM_dwimap.mif.gz"


def test_session_auto_picked_when_exactly_one_exists(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"
    dwi_dir = qsirecon_dir / subject / session / "dwi"
    _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    _add_dseg_and_labels(dwi_dir, subject, session)
    _add_act(qsirecon_dir / subject / "anat", subject, session)

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=None,
        atlas=ATLAS,
        workflow_hint=None,
        allow_missing_act=False,
    )

    assert found.wm_fod == dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz"


def test_session_required_when_multiple_exist_and_none_given(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject = "sub-01"
    for session in ("ses-01", "ses-02"):
        dwi_dir = qsirecon_dir / subject / session / "dwi"
        _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz")

    with pytest.raises(ValueError):
        discover_bundle(
            derivatives_dir=None,
            qsirecon_dir=qsirecon_dir,
            qsiprep_dir=None,
            subject=subject,
            session=None,
            atlas=ATLAS,
            workflow_hint=None,
            allow_missing_act=False,
        )


def test_act_tries_patterns_in_documented_order(tmp_path):
    """When only the 5tt-pattern file exists (no hsvs probseg), it is still found."""
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"
    dwi_dir = qsirecon_dir / subject / session / "dwi"
    _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    _add_dseg_and_labels(dwi_dir, subject, session)
    anat_dir = qsirecon_dir / subject / "anat"
    act_path = anat_dir / f"{subject}_{session}_5tt.mif.gz"
    _touch(act_path)

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=session,
        atlas=ATLAS,
        workflow_hint=None,
        allow_missing_act=False,
    )

    assert found.act_5tt_or_hsvs == act_path


def test_act_missing_raises_unless_allow_missing_act(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"
    dwi_dir = qsirecon_dir / subject / session / "dwi"
    _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    _add_dseg_and_labels(dwi_dir, subject, session)
    # No ACT files anywhere.

    with pytest.raises(FileNotFoundError):
        discover_bundle(
            derivatives_dir=None,
            qsirecon_dir=qsirecon_dir,
            qsiprep_dir=None,
            subject=subject,
            session=session,
            atlas=ATLAS,
            workflow_hint=None,
            allow_missing_act=False,
        )

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=session,
        atlas=ATLAS,
        workflow_hint=None,
        allow_missing_act=True,
    )
    assert found.act_5tt_or_hsvs is None


def test_parcellation_dseg_and_labels_resolved_for_given_atlas(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"
    dwi_dir = qsirecon_dir / subject / session / "dwi"
    _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    _add_act(qsirecon_dir / subject / "anat", subject, session)
    dseg_path = dwi_dir / f"{subject}_{session}_space-ACPC_seg-{ATLAS}_dseg.mif.gz"
    labels_path = dwi_dir / f"{subject}_{session}_seg-{ATLAS}_dseg.txt"
    _touch(dseg_path)
    _touch(labels_path)

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=session,
        atlas=ATLAS,
        workflow_hint=None,
        allow_missing_act=False,
    )

    assert found.dseg == dseg_path
    assert found.labels == labels_path


def test_labels_prefers_exact_txt_over_fuzzy_match(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"
    dwi_dir = qsirecon_dir / subject / session / "dwi"
    _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    _add_act(qsirecon_dir / subject / "anat", subject, session)
    dseg_path = dwi_dir / f"{subject}_{session}_space-ACPC_seg-{ATLAS}_dseg.mif.gz"
    _touch(dseg_path)

    exact_labels = dwi_dir / f"{subject}_{session}_seg-{ATLAS}_dseg.txt"
    fuzzy_labels = dwi_dir / f"{subject}_{session}_seg-{ATLAS}_dseg_extra.tsv"
    _touch(fuzzy_labels)
    _touch(exact_labels)

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=session,
        atlas=ATLAS,
        workflow_hint=None,
        allow_missing_act=False,
    )

    assert found.labels == exact_labels


def test_build_config_json_shape(tmp_path):
    qsirecon_dir = tmp_path / "qsirecon"
    subject, session = "sub-01", "ses-01"
    dwi_dir = qsirecon_dir / subject / session / "dwi"
    _touch(dwi_dir / f"{subject}_{session}_label-WM_dwimap.mif.gz")
    _add_dseg_and_labels(dwi_dir, subject, session)
    _add_act(qsirecon_dir / subject / "anat", subject, session)

    found = discover_bundle(
        derivatives_dir=None,
        qsirecon_dir=qsirecon_dir,
        qsiprep_dir=None,
        subject=subject,
        session=session,
        atlas=ATLAS,
        workflow_hint=None,
        allow_missing_act=False,
    )

    cfg = build_config_json(found, ATLAS)

    assert cfg["backend"] == "mrtrix"
    assert cfg["inputs"]["bundle"]["wm_fod"] == str(found.wm_fod)
    assert cfg["inputs"]["bundle"]["act_5tt_or_hsvs"] == str(found.act_5tt_or_hsvs)
    parcellations = cfg["inputs"]["bundle"]["parcellations"]
    assert parcellations == [
        {"name": ATLAS, "dseg": str(found.dseg), "labels_tsv": str(found.labels)}
    ]
    assert "mrtrix" in cfg
    assert "search_space" in cfg
