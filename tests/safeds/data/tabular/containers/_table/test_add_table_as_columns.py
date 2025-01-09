import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import DuplicateColumnError, RowCountMismatchError


@pytest.mark.parametrize(
    ("table", "other", "expected"),
    [
        (
            Table({}),
            Table({}),
            Table({}),
        ),
        (
            Table({}),
            Table({"col1": [1]}),
            Table({"col1": [1]}),
        ),
        (
            Table({"col1": [1]}),
            Table({}),
            Table({"col1": [1]}),
        ),
        (
            Table({"col1": [1]}),
            Table({"col2": [2]}),
            Table({"col1": [1], "col2": [2]}),
        ),
    ],
    ids=[
        "empty table, empty table",
        "empty table, non-empty table",
        "non-empty table, empty table",
        "non-empty table, non-empty table",
    ],
)
def test_should_add_columns(table: Table, other: Table, expected: Table) -> None:
    actual = table.add_table_as_columns(other)
    assert actual.schema == expected.schema
    assert actual == expected


def test_should_raise_error_if_row_counts_differ() -> None:
    with pytest.raises(RowCountMismatchError):
        Table({"col1": [1]}).add_table_as_columns(Table({"col2": [1, 2]}))


def test_should_raise_error_if_duplicate_column_name() -> None:
    with pytest.raises(DuplicateColumnError):
        Table({"col1": [1]}).add_table_as_columns(Table({"col1": [1]}))
