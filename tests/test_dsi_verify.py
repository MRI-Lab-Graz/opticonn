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
