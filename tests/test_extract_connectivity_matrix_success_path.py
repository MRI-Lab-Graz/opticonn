"""Regression test for a dead call left behind by an incomplete rename.

Commit 205a1e5 renamed ConnectivityExtractor._organize_output_files into
_check_connectivity_files_created but left the old call site in
extract_connectivity_matrix's success branch, so any atlas that actually
succeeds crashes with AttributeError before returning. This was masked for
months because a separate bug (fixed in test_dsi_studio_mat_compat.py) made
_check_connectivity_files_created return False for every real DSI Studio
run, so the success branch was never reached.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.extract_connectivity_matrices import ConnectivityExtractor


def test_extract_connectivity_matrix_success_path_does_not_raise(tmp_path):
    extractor = ConnectivityExtractor(config={"connectivity_values": ["fa"]})

    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with patch("scripts.extract_connectivity_matrices.subprocess.run", return_value=fake_result), \
         patch.object(extractor, "_check_connectivity_files_created", return_value=True):
        result = extractor.extract_connectivity_matrix(
            input_file="sub01.gqi.fz",
            output_dir=tmp_path,
            atlas="Atlas",
            base_name="sub01.gqi",
        )

    assert result["success"] is True
