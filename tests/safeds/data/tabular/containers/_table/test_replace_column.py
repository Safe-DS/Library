from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.exceptions import (
    ColumnNotFoundError,
    DuplicateColumnError,
    LengthMismatchError,
)


@pytest.mark.parametrize(
    ("table_factory", "old_name", "new_columns", "expected"),
    [
        (
            lambda: Table({"before": [], "col1": [], "after": []}),
            "col1",
            Column("col2", []),
            Table({"before": [], "col2": [], "after": []}),
        ),
        (
            lambda: Table({"before": [], "col1": [], "after": []}),
            "col1",
            [],
            Table({"before": [], "after": []}),
        ),
        (
            lambda: Table({"before": [], "col1": [], "after": []}),
            "col1",
            [Column("col3", []), Column("col4", [])],
            Table({"before": [], "col3": [], "col4": [], "after": []}),
        ),
        (
            lambda: Table({"before": [], "col1": [], "after": []}),
            "col1",
            Table({"col3": [], "col4": []}),
            Table({"before": [], "col3": [], "col4": [], "after": []}),
        ),
        (
            lambda: Table({"col1": []}),
            "col1",
            Column("col1", []),
            Table({"col1": []}),
        ),
    ],
    ids=[
        "single new column",
        "empty list of new columns",
        "non-empty list of new columns",
        "new table",
        "reusing the old name",
    ],
)
class TestHappyPath:
    def test_should_replace_column(
        self,
        table_factory: Callable[[], Table],
        old_name: str,
        new_columns: list[Column],
        expected: Table,
    ) -> None:
        actual = table_factory().replace_column(old_name, new_columns)
        assert actual.schema == expected.schema
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        old_name: str,
        new_columns: list[Column],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.replace_column(old_name, new_columns)
        assert original == table_factory()


def test_should_raise_if_old_name_is_unknown() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).replace_column("col1", [Column("a", [1, 2])])


@pytest.mark.parametrize(
    ("table", "old_name", "new_columns"),
    [
        (
            Table({"col1": [], "col2": []}),
            "col1",
            Column("col2", []),
        ),
        (
            Table({"col1": [], "col2": []}),
            "col1",
            [Column("col2", [])],
        ),
        (
            Table({"col1": [], "col2": []}),
            "col1",
            Table({"col2": []}),
        ),
        (
            Table({"col1": []}),
            "col1",
            [Column("col2", []), Column("col2", [])],
        ),
    ],
    ids=[
        "single new column",
        "list of new columns",
        "new table",
        "duplicate new names",
    ],
)
def test_should_raise_if_new_column_exists_already(
    table: Table,
    old_name: str,
    new_columns: Column | list[Column],
) -> None:
    with pytest.raises(DuplicateColumnError):
        table.replace_column(old_name, new_columns)


@pytest.mark.parametrize(
    ("table", "old_name", "new_columns"),
    [
        (
            Table({"col1": []}),
            "col1",
            Column("col2", [1]),
        ),
        (
            Table({"col1": []}),
            "col1",
            [Column("col2", [1])],
        ),
        (
            Table({"col1": []}),
            "col1",
            Table({"col2": [1]}),
        ),
    ],
    ids=[
        "single new column",
        "list of new columns",
        "new table",
    ],
)
def test_should_raise_if_row_counts_differ(
    table: Table,
    old_name: str,
    new_columns: Column | list[Column] | Table,
) -> None:
    with pytest.raises(LengthMismatchError):
        table.replace_column(old_name, new_columns)
