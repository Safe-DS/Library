import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import (
    ColumnSizeError,
    DuplicateColumnNameError,
    UnknownColumnNameError,
)


@pytest.mark.parametrize(
    ("table", "column_name", "columns", "expected"),
    [
        (
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "C": ["a", "b", "c"],
                },
            ),
            "B",
            [Column("B", ["d", "e", "f"]), Column("D", [3, 4, 5])],
            Table(
                {
                    "A": [1, 2, 3],
                    "B": ["d", "e", "f"],
                    "D": [3, 4, 5],
                    "C": ["a", "b", "c"],
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
            [Column("D", ["d", "e", "f"])],
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                    "D": ["d", "e", "f"],
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
            [],
            Table(
                {
                    "A": [1, 2, 3],
                    "B": [4, 5, 6],
                },
            ),
        ),
    ],
    ids=["multiple Columns", "one Column", "empty"],
)
def test_should_replace_column(table: Table, column_name: str, columns: list[Column], expected: Table) -> None:
    result = table.replace_column(column_name, columns)
    assert result._schema == expected._schema
    assert result == expected


@pytest.mark.parametrize(
    ("old_column_name", "column", "error", "error_message"),
    [
        ("D", [Column("C", ["d", "e", "f"])], UnknownColumnNameError, r"Could not find column\(s\) 'D'"),
        (
            "C",
            [Column("B", ["d", "e", "f"]), Column("D", [3, 2, 1])],
            DuplicateColumnNameError,
            r"Column 'B' already exists.",
        ),
        (
            "C",
            [Column("D", [7, 8]), Column("E", ["c", "b"])],
            ColumnSizeError,
            r"Expected a column of size 3 but got column of size 2.",
        ),
    ],
    ids=["UnknownColumnNameError", "DuplicateColumnNameError", "ColumnSizeError"],
)
def test_should_raise_error(
    old_column_name: str,
    column: list[Column],
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

    with pytest.raises(error, match=error_message):
        input_table.replace_column(old_column_name, column)


def test_should_fail_on_empty_table() -> None:
    with pytest.raises(UnknownColumnNameError):
        Table().replace_column("col", [Column("a", [1, 2])])
