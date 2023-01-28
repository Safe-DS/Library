import pandas as pd
import pytest
from safeds.data import Column, Table
from safeds.exceptions import (
    ColumnSizeError,
    DuplicateColumnNameError,
    UnknownColumnNameError,
)


@pytest.mark.parametrize(
    "column_name, path",
    [
        ("C", "tests/resources/test_table_replace_column_output_different_name.csv"),
        ("D", "tests/resources/test_table_replace_column_output_same_name.csv"),
    ],
)
def test_replace_valid(column_name: str, path: str) -> None:
    input_table: Table = Table.from_csv(
        "tests/resources/test_table_replace_column_input.csv"
    )
    expected: Table = Table.from_csv(path)

    column = Column(pd.Series(["d", "e", "f"]), column_name)

    result = input_table.replace_column("C", column)

    assert result == expected


@pytest.mark.parametrize(
    "old_column_name, column_values, column_name, error",
    [
        ("D", ["d", "e", "f"], "C", UnknownColumnNameError),
        ("C", ["d", "e", "f"], "B", DuplicateColumnNameError),
        ("C", ["d", "e"], "D", ColumnSizeError),
    ],
)
def test_replace_invalid(
    old_column_name: str,
    column_values: list[str],
    column_name: str,
    error: type[Exception],
) -> None:
    input_table: Table = Table.from_csv(
        "tests/resources/test_table_replace_column_input.csv"
    )
    column = Column(pd.Series(column_values), column_name)

    with pytest.raises(error):
        input_table.replace_column(old_column_name, column)
