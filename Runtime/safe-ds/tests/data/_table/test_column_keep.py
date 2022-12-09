import pytest
from safe_ds.data import Table
from safe_ds.exceptions import UnknownColumnNameError


def test_table_column_keep() -> None:
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    transformed_table = table.keep_columns(["A"])
    assert (
        "A" in transformed_table._data.columns
        and "B" not in transformed_table._data.columns
    )


def test_table_column_keep_warning() -> None:
    table = Table.from_csv("tests/resources/test_table_read_csv.csv")
    with pytest.raises(UnknownColumnNameError):
        table.keep_columns(["C"])
