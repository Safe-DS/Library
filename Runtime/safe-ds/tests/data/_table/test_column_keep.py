import pytest
from safe_ds.data import Table
from safe_ds.exceptions import ColumnNameError


def test_table_column_keep():
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    transformed_table = table.keep_columns(["A"])
    assert (
        "A" in transformed_table._data.columns
        and "B" not in transformed_table._data.columns
    )


def test_table_column_keep_warning():
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    with pytest.raises(ColumnNameError):
        table.keep_columns(["C"])
