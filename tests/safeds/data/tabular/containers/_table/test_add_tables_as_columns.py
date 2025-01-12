from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import DuplicateColumnError, LengthMismatchError


@pytest.mark.parametrize(
    ("table_factory", "others_factory", "expected"),
    [
        (
            lambda: Table({}),
            lambda: Table({}),
            Table({}),
        ),
        (
            lambda: Table({}),
            lambda: Table({"col1": [1]}),
            Table({"col1": [1]}),
        ),
        (
            lambda: Table({"col1": [1]}),
            lambda: Table({}),
            Table({"col1": [1]}),
        ),
        (
            lambda: Table({"col1": [1]}),
            lambda: Table({"col2": [2]}),
            Table({"col1": [1], "col2": [2]}),
        ),
        (
            lambda: Table({"col1": [1]}),
            lambda: [
                Table({"col2": [2]}),
                Table({"col3": [3]}),
            ],
            Table({"col1": [1], "col2": [2], "col3": [3]}),
        ),
    ],
    ids=[
        "empty, empty",
        "empty, non-empty",
        "non-empty, empty",
        "non-empty, non-empty",
        "multiple tables",
    ],
)
class TestHappyPath:
    def test_should_add_columns(
        self,
        table_factory: Callable[[], Table],
        others_factory: Callable[[], Table | list[Table]],
        expected: Table,
    ) -> None:
        actual = table_factory().add_tables_as_columns(others_factory())
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        others_factory: Callable[[], Table | list[Table]],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.add_tables_as_columns(others_factory())
        assert original == table_factory()

    def test_should_not_mutate_others(
        self,
        table_factory: Callable[[], Table],
        others_factory: Callable[[], Table | list[Table]],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = others_factory()
        table_factory().add_tables_as_columns(original)
        assert original == others_factory()


@pytest.mark.parametrize(
    ("table", "others"),
    [
        (
            Table({"col1": [1]}),
            Table({"col2": [1, 2]}),
        ),
        (
            Table({"col1": [1]}),
            [
                Table({"col2": [1]}),
                Table({"col3": [1, 2]}),
            ],
        ),
    ],
    ids=[
        "one table",
        "multiple tables",
    ],
)
def test_should_raise_if_row_counts_differ(table: Table, others: Table | list[Table]) -> None:
    with pytest.raises(LengthMismatchError):
        table.add_tables_as_columns(others)


@pytest.mark.parametrize(
    ("table", "others"),
    [
        (
            Table({"col1": [1]}),
            Table({"col1": [1]}),
        ),
        (
            Table({"col1": [1]}),
            [
                Table({"col2": [1]}),
                Table({"col1": [1]}),
            ],
        ),
        (
            Table({"col1": [1]}),
            [
                Table({"col2": [1]}),
                Table({"col2": [1]}),
            ],
        ),
    ],
    ids=[
        "one table",
        "multiple tables, clash with receiver",
        "multiple tables, clash with other table",
    ],
)
def test_should_raise_if_duplicate_column_name(table: Table, others: Table | list[Table]) -> None:
    with pytest.raises(DuplicateColumnError):
        table.add_tables_as_columns(others)
