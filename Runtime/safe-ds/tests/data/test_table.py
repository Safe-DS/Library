import pytest
from safe_ds.data import Table
from safe_ds.exceptions import ColumnNameDuplicateError, ColumnNameError


def test_read_csv_valid():
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    assert table._data["A"][0] == 1 and table._data["B"][0] == 2


def test_read_csv_invalid():
    with pytest.raises(FileNotFoundError):
        Table.from_csv("tests/resources/test_table_read_csv_invalid.csv")


def test_read_json_valid():
    table = Table.from_json("tests/resources/test_table_read_json.json")
    assert table._data["A"][0] == 1 and table._data["B"][0] == 2


def test_read_json_invalid():
    with pytest.raises(FileNotFoundError):
        Table.from_json("tests/resources/test_table_read_json_invalid.json")


def test_get_row_by_index():
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    val = table.get_row_by_index(0)
    assert val._data["A"] == 1 and val._data["B"] == 2


def test_get_row_by_index_negative_index():
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    with pytest.raises(KeyError):
        table.get_row_by_index(-1)


def test_get_row_by_index_out_of_bounds_index():
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    with pytest.raises(KeyError):
        table.get_row_by_index(5)


@pytest.mark.parametrize(
    "name_from, name_to, column_one, column_two",
    [("A", "D", "D", "B"), ("A", "A", "A", "B")],
)
def test_rename_valid(name_from, name_to, column_one, column_two):
    table: Table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    renamed_table = table.rename_column(name_from, name_to)
    assert renamed_table._data.columns[0] == column_one
    assert renamed_table._data.columns[1] == column_two


@pytest.mark.parametrize(
    "name_from, name_to, error",
    [
        ("C", "D", ColumnNameError),
        ("A", "B", ColumnNameDuplicateError),
        ("D", "D", ColumnNameError),
    ],
)
def test_rename_invalid(name_from, name_to, error):
    table: Table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    with pytest.raises(error):
        table.rename_column(name_from, name_to)
