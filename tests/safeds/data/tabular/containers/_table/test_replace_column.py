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
            Column("D", ["d", "e", "f"]),
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
            "B",
            Table(
                {
                    "D": [7, 8, 9],
                    "E": ["c", "b", "a"],
                },
            ),
            Table(
                {
                    "A": [1, 2, 3],
                    "D": [7, 8, 9],
                    "E": ["c", "b", "a"],
                    "C": ["a", "b", "c"],
                },
            ),
        ),
    ],
    ids=["list[Column]", "Column", "Table"],
)
def test_should_replace_column(table: Table, column_name: str, columns: Column | list[Column] | Table, expected: Table) -> None:
    result = table.replace_column(column_name, columns)
    assert result == expected


@pytest.mark.parametrize(
    ("old_column_name", "column", "error"),
    [
        ("D", Column("C", ["d", "e", "f"]), UnknownColumnNameError),
        ("C", [Column("B", ["d", "e", "f"]), Column("D", [3, 2, 1])], DuplicateColumnNameError),
        ("C", Table({"D": [7, 8], "E": ["c", "b"]}), ColumnSizeError),
    ],
    ids=["UnknownColumnNameError", "DuplicateColumnNameError", "ColumnSizeError"],
)
def test_should_raise_error(
    old_column_name: str,
    column: Column | list[Column] | Table,
    error: type[Exception],
) -> None:
    input_table: Table = Table(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": ["a", "b", "c"],
        },
    )

    with pytest.raises(error):
        input_table.replace_column(old_column_name, column)
