import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import (
    ColumnSizeError,
    DuplicateColumnNameError,
    UnknownColumnNameError,
)


@pytest.mark.parametrize(
    ("table", "column_name", "column", "expected"),
    [
        (
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["a", "b", "c"],
                },
            ),
            "C",
            Column("C", ["d", "e", "f"]),
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["d", "e", "f"],
                },
            ),
        ),
        (
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["a", "b", "c"],
                },
            ),
            "C",
            Column("D", ["d", "e", "f"]),
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "D": ["d", "e", "f"],
                },
            ),
        ),
    ],
)
def test_should_replace_column(table: Table, column_name: str, column: Column, expected: Table) -> None:
    result = table.replace_column(column_name, column)
    assert result == expected


@pytest.mark.parametrize(
    ("old_column_name", "column_values", "column_name", "error", "error_message"),
    [
        ("D", ["d", "e", "f"], "C", UnknownColumnNameError, r"Could not find column\(s\) 'D'"),
        ("C", ["d", "e", "f"], "B", DuplicateColumnNameError, r"Column 'B' already exists."),
        ("C", ["d", "e"], "D", ColumnSizeError, r"Expected a column of size 3 but got column of size 2."),
    ],
    ids=["UnknownColumnNameError", "DuplicateColumnNameError", "ColumnSizeError"],
)
def test_should_raise_error(
    old_column_name: str,
    column_values: list[str],
    column_name: str,
    error: type[Exception],
    error_message: str,
) -> None:
    input_table: Table = Table(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": ["a", "b", "c"],
        },
    )
    column = Column(column_name, column_values)

    with pytest.raises(error, match=error_message):
        input_table.replace_column(old_column_name, column)
