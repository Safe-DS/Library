import pytest
from safe_ds.data import Table
from safe_ds.exceptions import UnknownColumnNameError


def test_transform_column_valid() -> None:
    input_table: Table = Table.from_csv(
        "tests/resources/test_table_transform_column.csv"
    )

    result: Table = input_table.transform_column(
        "A", lambda row: row.get_value("A") * 2
    )

    assert result == Table.from_csv(
        "tests/resources/test_table_transform_column_output.csv"
    )


def test_transform_column_invalid() -> None:
    input_table: Table = Table.from_csv(
        "tests/resources/test_table_transform_column.csv"
    )

    with pytest.raises(UnknownColumnNameError):
        input_table.transform_column("D", lambda row: row.get_value("A") * 2)
