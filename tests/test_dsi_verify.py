from pathlib import Path

import pytest

from scripts import dsi_verify

FIX = Path(__file__).parent / "fixtures" / "dsi_studio_echo"
WARNING = "--connectivity_threshold is not used/recognized"


def _stdout(name="sweep_0001"):
    return (FIX / f"{name}_stdout.txt").read_text()


def _sent(name="sweep_0001"):
    return dsi_verify.parse_command_flags((FIX / f"{name}_command.txt").read_text().split())


def _without_warning(text):
    return "\n".join(line for line in text.splitlines() if WARNING not in line)


def test_parse_command_flags_reads_every_flag():
    sent = _sent()
    assert sent["action"] == "trk"
    assert sent["turning_angle"] == "35"
    assert sent["connectivity_threshold"] == "0.001"
    assert all(not k.startswith("-") for k in sent)


def test_parse_echo_strips_ansi_and_tree_prefixes():
    echo = dsi_verify.parse_echo(_stdout())
    assert echo["action"] == "trk"
    assert echo["fa_threshold"] == "0"
    assert echo["turning_angle"] == "35"
    assert echo["track_voxel_ratio"] == "0.738525"   # derived by DSI Studio, never sent
    assert "connectivity_threshold" not in echo


def test_unrecognised_flag_fails_twice_over():
    errors = dsi_verify.check_echo(_sent(), _stdout())
    assert any("connectivity_threshold" in e and "not used/recognized" in e for e in errors)
    assert any("connectivity_threshold" in e and "not echoed back" in e for e in errors)
    assert all("connectivity_threshold" in e for e in errors), errors


def test_sent_flag_missing_from_echo_fails_even_without_the_warning():
    # A build that stops printing the warning must still be caught.
    errors = dsi_verify.check_echo(_sent(), _without_warning(_stdout()))
    assert errors == [
        "--connectivity_threshold=0.001: sent but not echoed back, so not executed"]


def test_faithful_echo_passes():
    sent = _sent()
    del sent["connectivity_threshold"]
    assert dsi_verify.check_echo(sent, _without_warning(_stdout())) == []


def test_changed_value_fails():
    sent = _sent()
    del sent["connectivity_threshold"]
    sent["turning_angle"] = "40"
    errors = dsi_verify.check_echo(sent, _without_warning(_stdout()))
    assert errors == ["--turning_angle: sent 40, DSI Studio parsed 35"]


def test_numeric_formatting_is_not_a_mismatch():
    sent = _sent()
    del sent["connectivity_threshold"]
    sent["step_size"] = "1"          # echo says 1.0
    assert dsi_verify.check_echo(sent, _without_warning(_stdout())) == []


def test_paths_are_confirmed_present_but_not_compared():
    sent = _sent()
    del sent["connectivity_threshold"]
    sent["source"] = "/somewhere/else.fz"
    assert dsi_verify.check_echo(sent, _without_warning(_stdout())) == []


def test_no_echo_at_all_fails():
    assert dsi_verify.check_echo({"action": "trk"}, "") == [
        "no parameter echo found in DSI Studio output; cannot confirm anything was executed"]


def _report(name):
    return (FIX / f"exec_{name}_report.txt").read_text()


GRID_SENT = {"turning_angle": "35", "step_size": "1.0", "smoothing": "0.1",
             "min_length": "10", "max_length": "250", "tract_count": "50000",
             "otsu_threshold": "0.6", "fa_threshold": "0.0"}


def test_parse_report_grid():
    ex = dsi_verify.parse_report(_report("grid"))
    assert ex["anisotropy"] == {"kind": "otsu_window", "low": 0.5, "high": 0.7}
    assert ex["turning_angle"] == {"kind": "fixed", "value": 35.0}
    assert ex["step_size"] == {"kind": "fixed", "value": 1.0}
    assert ex["smoothing"] == pytest.approx(0.1)
    assert (ex["min_length"], ex["max_length"]) == (10.0, 250.0)
    assert ex["tract_count"] == 50000
    assert ex["tip_iteration"] == 0


def test_parse_report_reference_resolves_defaults():
    ex = dsi_verify.parse_report(_report("reference"))
    assert ex["anisotropy"] == {"kind": "otsu_window", "low": 0.5, "high": 0.7}
    assert ex["turning_angle"] == {"kind": "random", "low": 45.0, "high": 90.0}
    assert ex["step_size"] == {"kind": "voxel_spacing"}
    assert ex["smoothing"] == 0.0
    assert (ex["min_length"], ex["max_length"]) == (30.0, 200.0)


def test_parse_report_fixed_fa_otsu_window_and_pruning():
    assert dsi_verify.parse_report(_report("fa01"))["anisotropy"] == {"kind": "fixed", "value": 0.1}
    assert dsi_verify.parse_report(_report("otsu08"))["anisotropy"] == {
        "kind": "otsu_window", "low": 0.7, "high": 0.9}
    assert dsi_verify.parse_report(_report("tip2"))["tip_iteration"] == 2


def test_grid_report_matches_what_was_sent():
    executed, errors = dsi_verify.check_report(GRID_SENT, _report("grid"))
    assert errors == []
    assert executed["tract_count"] == 50000


def test_otsu_window_is_derived_from_the_sent_centre():
    sent = {**GRID_SENT, "otsu_threshold": "0.8"}
    _, errors = dsi_verify.check_report(sent, _report("grid"))
    assert len(errors) == 1 and errors[0].startswith("anisotropy:")


def test_fixed_fa_threshold_takes_precedence_over_otsu():
    sent = {**GRID_SENT, "fa_threshold": "0.1"}          # otsu 0.6 still sent, inert
    assert dsi_verify.check_report(sent, _report("fa01"))[1] == []


def test_zero_angle_means_the_documented_random_window():
    sent = {"turning_angle": "0"}
    assert dsi_verify.check_report(sent, _report("reference"))[1] == []
    assert dsi_verify.check_report(sent, _report("grid"))[1] != []


def test_zero_step_means_voxel_spacing():
    assert dsi_verify.check_report({"step_size": "0"}, _report("reference"))[1] == []


def test_smoothing_sent_but_not_executed_fails():
    _, errors = dsi_verify.check_report({"smoothing": "0.1"}, _report("reference"))
    assert errors == ["smoothing: expected 0.1, DSI Studio executed 0.0"]


def test_pruning_is_verified():
    assert dsi_verify.check_report({"tip_iteration": "2"}, _report("tip2"))[1] == []
    assert dsi_verify.check_report({"tip_iteration": "2"}, _report("grid"))[1] != []


def test_changed_length_fails():
    _, errors = dsi_verify.check_report({**GRID_SENT, "min_length": "30"}, _report("grid"))
    assert errors == ["min_length: expected 30.0, DSI Studio executed 10.0"]


def test_nothing_sent_means_nothing_compared_but_everything_recorded():
    executed, errors = dsi_verify.check_report({}, _report("reference"))
    assert errors == []
    assert executed["step_size"] == {"kind": "voxel_spacing"}


def test_empty_or_reworded_report_fails_closed():
    _, errors = dsi_verify.check_report(GRID_SENT, "")
    assert {e.split(" for ")[-1] for e in errors} >= {
        "anisotropy", "turning_angle", "step_size", "min_length", "tract_count"}


import gzip
import io

import numpy as np
import scipy.io


def _pid(name):
    return (FIX / f"exec_{name}_parameter_id.txt").read_text().strip()


def _write_tract(path, report, parameter_id, track=b"\x01\x02\x03"):
    buf = io.BytesIO()
    as_u8 = lambda s: np.frombuffer(s.encode() if isinstance(s, str) else s, dtype=np.uint8)
    scipy.io.savemat(buf, {"report": as_u8(report), "parameter_id": as_u8(parameter_id),
                           "track": as_u8(track)})
    path.write_bytes(gzip.compress(buf.getvalue()))


def test_read_tract_record_roundtrip(tmp_path):
    p = tmp_path / "x.tt.gz"
    _write_tract(p, _report("grid"), _pid("grid"))
    rec = dsi_verify.read_tract_record(p)
    assert rec["parameter_id"] == _pid("grid")
    assert "angular threshold was 35 degrees" in rec["report"]
    assert len(rec["track_sha256"]) == 64


def test_fingerprints_differ_for_every_varied_parameter():
    ids = {n: _pid(n) for n in ["grid", "rk4", "tip2", "otsu08", "fa01", "reference"]}
    assert len(set(ids.values())) == len(ids), ids


def test_fingerprint_key_ignores_paths_and_order():
    a = {"source": "/a.fz", "output": "/o.tt.gz", "connectivity": "/atlas.nii.gz",
         "turning_angle": "35", "random_seed": "1"}
    b = {"random_seed": "1", "turning_angle": "35", "source": "/b.fz",
         "output": "/p.tt.gz", "connectivity": "/other.nii.gz"}
    assert dsi_verify.fingerprint_key(a) == dsi_verify.fingerprint_key(b)
    assert dsi_verify.fingerprint_key({**a, "random_seed": "2"}) != dsi_verify.fingerprint_key(a)


def test_check_fingerprint():
    sent = {"turning_angle": "35", "random_seed": "1"}
    key = dsi_verify.fingerprint_key(sent)
    assert dsi_verify.check_fingerprint(sent, _pid("grid"), {key: _pid("grid")}) == []
    assert dsi_verify.check_fingerprint(sent, _pid("rk4"), {key: _pid("grid")})[0].startswith(
        "parameter_id")
    assert "run the preflight" in dsi_verify.check_fingerprint(sent, _pid("grid"), {})[0]


# Geometry: synthetic streamlines in mm, executed parameters as parsed from a report.
EXEC = {"step_size": {"kind": "fixed", "value": 1.0},
        "turning_angle": {"kind": "fixed", "value": 35.0},
        "min_length": 10.0, "max_length": 250.0, "tract_count": 2}


def _line(n_points, step=1.0):
    return np.column_stack([np.arange(n_points) * step, np.zeros(n_points), np.zeros(n_points)])


def test_geometry_passes_on_compliant_streamlines():
    assert dsi_verify.check_geometry([_line(10), _line(40)], EXEC, voxel_size=1.7) == []


def test_geometry_uses_dsi_studio_length_convention():
    # 10 points at 1 mm: segment sum 9 mm, DSI Studio counts 10 mm -> kept at min 10.
    assert dsi_verify.check_geometry([_line(10), _line(11)], EXEC, voxel_size=1.7) == []
    assert dsi_verify.check_geometry([_line(9), _line(11)], EXEC, voxel_size=1.7) != []


def test_geometry_catches_wrong_step_count_and_turn():
    assert any("step" in e for e in dsi_verify.check_geometry(
        [_line(20, step=0.5), _line(20)], EXEC, voxel_size=1.7))
    assert any("streamlines" in e for e in dsi_verify.check_geometry(
        [_line(20)], EXEC, voxel_size=1.7))
    turned = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                       [2 + np.cos(np.radians(40)), np.sin(np.radians(40)), 0]])
    turned = np.vstack([turned, turned[-1] + np.arange(1, 10)[:, None] * [np.cos(np.radians(40)),
                                                                         np.sin(np.radians(40)), 0]])
    assert any("turn" in e for e in dsi_verify.check_geometry(
        [turned, _line(20)], EXEC, voxel_size=1.7))


def test_geometry_voxel_spacing_step_and_random_angle():
    ex = {**EXEC, "step_size": {"kind": "voxel_spacing"},
          "turning_angle": {"kind": "random", "low": 45.0, "high": 90.0},
          "min_length": 30.0, "max_length": 200.0}
    assert dsi_verify.check_geometry([_line(30, 1.7), _line(40, 1.7)], ex, voxel_size=1.7) == []


def test_geometry_length_uses_whole_step_counts_not_points_times_step():
    # Real preflight numbers (2026-09-27): step 1.71875 mm, min 30 / max 200 mm ->
    # floor(30/1.71875)=17, floor(200/1.71875)=116. Both bounds are exact, so a
    # streamline one point short or long of either must fail.
    ex = {**EXEC, "step_size": {"kind": "fixed", "value": 1.71875},
          "min_length": 30.0, "max_length": 200.0}
    assert dsi_verify.check_geometry([_line(17, 1.71875), _line(116, 1.71875)],
                                     {**ex, "tract_count": 2}, voxel_size=1.71875) == []
    assert dsi_verify.check_geometry([_line(16, 1.71875), _line(116, 1.71875)],
                                     {**ex, "tract_count": 2}, voxel_size=1.71875) != []
    assert dsi_verify.check_geometry([_line(17, 1.71875), _line(117, 1.71875)],
                                     {**ex, "tract_count": 2}, voxel_size=1.71875) != []


def test_same_streamlines_tolerates_quantisation_only():
    a = [_line(10), _line(20)]
    assert dsi_verify.same_streamlines(a, [s + 0.02 for s in a]) == []
    assert dsi_verify.same_streamlines(a, [s + 0.2 for s in a]) != []
    assert dsi_verify.same_streamlines(a, a[:1]) != []
