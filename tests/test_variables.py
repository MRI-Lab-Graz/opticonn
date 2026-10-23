# tests/test_variables.py
import pytest

from scripts.variables import load_variables, subject_id


def _write(tmp_path, text, name="participants.tsv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_ids_match_with_and_without_the_sub_prefix(tmp_path):
    p = _write(tmp_path, "participant_id\tage\n01\t20\nsub-02\t30\n")
    table, _ = load_variables(p, ["age"])
    assert table == {"sub-01": {"age": 20.0}, "sub-02": {"age": 30.0}}
    assert subject_id("sub-7") == "sub-7" and subject_id(" 7 ") == "sub-7"


def test_missing_values_become_none(tmp_path):
    p = _write(tmp_path, "participant_id\tage\n01\tn/a\n02\t\n03\t41\n")
    table, _ = load_variables(p, ["age"])
    assert table["sub-01"]["age"] is None and table["sub-02"]["age"] is None
    assert table["sub-03"]["age"] == 41.0


def test_a_two_level_column_is_coded_in_sorted_order_and_the_coding_is_recorded(tmp_path):
    p = _write(tmp_path, "participant_id\tsex\n01\tM\n02\tF\n03\tM\n")
    table, coding = load_variables(p, ["sex"])
    assert [table[s]["sex"] for s in ("sub-01", "sub-02", "sub-03")] == [1.0, 0.0, 1.0]
    assert coding == {"sex": {"0": "F", "1": "M"}}


def test_a_numeric_two_valued_column_is_left_as_it_is(tmp_path):
    p = _write(tmp_path, "participant_id\tgroup\n01\t1\n02\t0\n")
    table, coding = load_variables(p, ["group"])
    assert table["sub-01"]["group"] == 1.0 and coding == {}


def test_more_than_two_levels_is_an_error_naming_the_column(tmp_path):
    p = _write(tmp_path, "participant_id\tsite\n01\tA\n02\tB\n03\tC\n")
    with pytest.raises(ValueError, match="site.*3 levels"):
        load_variables(p, ["site"])


def test_an_unknown_column_lists_the_available_ones(tmp_path):
    p = _write(tmp_path, "participant_id\tage\n01\t20\n")
    with pytest.raises(ValueError, match="no column named height.*age"):
        load_variables(p, ["height"])


def test_participant_id_is_required(tmp_path):
    p = _write(tmp_path, "id\tage\n01\t20\n")
    with pytest.raises(ValueError, match="participant_id"):
        load_variables(p, ["age"])


def test_a_comma_separated_file_is_read_too(tmp_path):
    p = _write(tmp_path, "participant_id,age\n01,20\n", name="vars.csv")
    table, _ = load_variables(p, ["age"])
    assert table == {"sub-01": {"age": 20.0}}


def test_a_repeated_participant_id_is_an_error(tmp_path):
    path = _write(tmp_path, "participant_id\tage\n01\t20\nsub-01\t30\n")
    with pytest.raises(ValueError, match="sub-01 appears more than once"):
        load_variables(path, ["age"])
