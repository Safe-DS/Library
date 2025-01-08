import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import DuplicateColumnError, RowCountMismatchError


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ([], Table({})),
        (Column("A", []), Table({"A": []})),
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
        "empty list",
        "single column",
        "non-empty list",
    ],
)
def test_should_create_table_from_columns(columns: Column | list[Column], expected: Table) -> None:
    assert Table.from_columns(columns) == expected


def test_should_raise_error_if_column_lengths_mismatch() -> None:
    with pytest.raises(RowCountMismatchError):
        Table.from_columns([Column("col1", []), Column("col2", [1])])


def test_should_raise_error_if_duplicate_column_name() -> None:
    with pytest.raises(DuplicateColumnError):
        Table.from_columns([Column("col1", []), Column("col1", [])])
