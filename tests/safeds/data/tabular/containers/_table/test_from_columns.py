import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import ColumnLengthMismatchError, DuplicateColumnError


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ([], Table()),
        (
            [
                Column("A", [1, 2]),
                Column("B", [3, 4]),
            ],
            Table(
                {
                    "A": [1, 2],
                    "B": [3, 4],
                },
            ),
        ),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_create_table_from_list_of_columns(columns: list[Column], expected: Table) -> None:
    assert Table.from_columns(columns) == expected


def test_should_raise_error_if_column_lengths_mismatch() -> None:
    with pytest.raises(ColumnLengthMismatchError):
        Table.from_columns([Column("col1", []), Column("col2", [1])])


def test_should_raise_error_if_duplicate_column_name() -> None:
    with pytest.raises(DuplicateColumnError):
        Table.from_columns([Column("col1", []), Column("col1", [])])
