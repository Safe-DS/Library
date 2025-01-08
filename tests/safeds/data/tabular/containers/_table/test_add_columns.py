import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import DuplicateColumnError, RowCountMismatchError


@pytest.mark.parametrize(
    ("table", "columns", "expected"),
    [
        (
            Table({}),
            [],
            Table({}),
        ),
        (
            Table({}),
            Column("col1", [1]),
            Table({"col1": [1]}),
        ),
        (
            Table({}),
            Table({"col1": [1]}),
            Table({"col1": [1]}),
        ),
        (
            Table({}),
            [Column("col1", [1]), Column("col2", [2])],
            Table({"col1": [1], "col2": [2]}),
        ),
        (
            Table({"col0": [0]}),
            [],
            Table({"col0": [0]}),
        ),
        (
            Table({"col0": [0]}),
            Column("col1", [1]),
            Table({"col0": [0], "col1": [1]}),
        ),
        (
            Table({"col0": [0]}),
            Table({"col1": [1]}),
            Table({"col0": [0], "col1": [1]}),
        ),
        (
            Table({"col0": [0]}),
            [Column("col1", [1]), Column("col2", [2])],
            Table({"col0": [0], "col1": [1], "col2": [2]}),
        ),
    ],
    ids=[
        "empty table, empty list",
        "empty table, single column",
        "empty table, single table",
        "empty table, multiple columns",
        "non-empty table, empty list",
        "non-empty table, single column",
        "empty table, single table",
        "non-empty table, multiple columns",
    ],
)
def test_should_add_columns(table: Table, columns: Column | list[Column] | Table, expected: Table) -> None:
    table = table.add_columns(columns)
    assert table.schema == expected.schema
    assert table == expected


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        (
            Table({"col1": [1]}),
            Column("col2", [1, 2]),
        ),
        (
            Table({"col1": [1]}),
            [Column("col2", [1, 2])],
        ),
        (
            Table({"col1": [1]}),
            Table({"col2": [1, 2]}),
        ),
    ],
    ids=[
        "single new column",
        "list of new columns",
        "table of new columns",
    ],
)
def test_should_raise_error_if_row_counts_differ(table: Table, columns: Column | list[Column] | Table) -> None:
    with pytest.raises(RowCountMismatchError):
        table.add_columns(columns)


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        (
            Table({"col1": [1]}),
            Column("col1", [1]),
        ),
        (
            Table({"col1": [1]}),
            [Column("col1", [1])],
        ),
        (
            Table({"col1": [1]}),
            Table({"col1": [1]}),
        ),
        (
            Table({}),
            [Column("col1", [1]), Column("col1", [2])],
        ),
    ],
    ids=[
        "single new column clashes with existing column",
        "list of new columns clashes with existing column",
        "table of new columns clashes with existing column",
        "new columns clash with each",
    ],
)
def test_should_raise_error_if_duplicate_column_name(table: Table, columns: Column | list[Column] | Table) -> None:
    with pytest.raises(DuplicateColumnError):
        table.add_columns(columns)
