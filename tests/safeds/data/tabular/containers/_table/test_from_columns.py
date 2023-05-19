import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import ColumnLengthMismatchError, DuplicateColumnNameError


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (
            [
                Column("A", [1, 4]),
                Column("B", [2, 5]),
            ],
            Table(
                {
                    "A": [1, 4],
                    "B": [2, 5],
                },
            ),
        ),
        ([], Table()),
    ],
    ids=["2 Columns", "empty"],
)
def test_should_create_table_from_list_of_columns(columns: list[Column], expected: Table) -> None:
    assert Table.from_columns(columns) == expected


def test_should_raise_error_if_column_length_mismatch() -> None:
    with pytest.raises(
        ColumnLengthMismatchError,
        match=r"The length of at least one column differs: \ncol1: 3\ncol2: 4",
    ):
        Table.from_columns([Column("col1", [5, 2, 3]), Column("col2", [5, 3, 4, 1])])


def test_should_raise_error_if_duplicate_column_name() -> None:
    with pytest.raises(DuplicateColumnNameError, match=r"Column 'col1' already exists."):
        Table.from_columns([Column("col1", [5, 2, 3]), Column("col1", [5, 3, 4])])
