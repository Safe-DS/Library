from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import DuplicateColumnError, LengthMismatchError


@pytest.mark.parametrize(
    ("table_factory", "columns", "expected"),
    [
        (
            lambda: Table({}),
            [],
            Table({}),
        ),
        (
            lambda: Table({}),
            Column("col1", [1]),
            Table({"col1": [1]}),
        ),
        (
            lambda: Table({}),
            Table({"col1": [1]}),
            Table({"col1": [1]}),
        ),
        (
            lambda: Table({}),
            [Column("col1", [1]), Column("col2", [2])],
            Table({"col1": [1], "col2": [2]}),
        ),
        (
            lambda: Table({"col0": [0]}),
            [],
            Table({"col0": [0]}),
        ),
        (
            lambda: Table({"col0": [0]}),
            Column("col1", [1]),
            Table({"col0": [0], "col1": [1]}),
        ),
        (
            lambda: Table({"col0": [0]}),
            Table({"col1": [1]}),
            Table({"col0": [0], "col1": [1]}),
        ),
        (
            lambda: Table({"col0": [0]}),
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
class TestHappyPath:
    def test_should_add_columns(
        self,
        table_factory: Callable[[], Table],
        columns: Column | list[Column] | Table,
        expected: Table,
    ) -> None:
        actual = table_factory().add_columns(columns)
        assert actual.schema == expected.schema
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        columns: Column | list[Column] | Table,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.add_columns(columns)
        assert original == table_factory()


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
    with pytest.raises(LengthMismatchError):
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
