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
