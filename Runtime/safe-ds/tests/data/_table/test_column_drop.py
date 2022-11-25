import pytest
from safe_ds.data import Table
from safe_ds.exceptions import ColumnNameError


def test_table_column_drop():
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    transformed_table = table.drop_columns(["A"])
    assert (
        "B" in transformed_table._data.columns
        and "A" not in transformed_table._data.columns
    )


def test_table_column_drop_warning():
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    with pytest.raises(ColumnNameError):
        table.drop_columns(["C"])
