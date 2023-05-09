import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import ColumnLengthMismatchError
from safeds.data.tabular.typing import Integer, Schema


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (
            [
                Column("A", [1, 4]),
                Column("B", [2, 5]),
            ],
            Table([[1, 2], [4, 5]], Schema({"A": Integer(), "B": Integer()})),
        ),
        ([], Table([])),
    ],
    ids=["2 Columns", "empty"],
)
def test_should_create_table_from_list_of_columns(columns: list[Column], expected: Table) -> None:
    assert Table.from_columns(columns) == expected


def test_should_raise_error_if_column_length_mismatch() -> None:
    with pytest.raises(ColumnLengthMismatchError):
        Table.from_columns([Column("col1", [5, 2, 3]), Column("col2", [5, 3, 4, 1])])
