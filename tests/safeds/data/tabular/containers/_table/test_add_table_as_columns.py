from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import DuplicateColumnError, LengthMismatchError


@pytest.mark.parametrize(
    ("table_factory", "other", "expected"),
    [
        (
            lambda: Table({}),
            Table({}),
            Table({}),
        ),
        (
            lambda: Table({}),
            Table({"col1": [1]}),
            Table({"col1": [1]}),
        ),
        (
            lambda: Table({"col1": [1]}),
            Table({}),
            Table({"col1": [1]}),
        ),
        (
            lambda: Table({"col1": [1]}),
            Table({"col2": [2]}),
            Table({"col1": [1], "col2": [2]}),
        ),
    ],
    ids=[
        "empty, empty",
        "empty, non-empty",
        "non-empty, empty",
        "non-empty, non-empty",
    ],
)
class TestHappyPath:
    def test_should_add_columns(
        self,
        table_factory: Callable[[], Table],
        other: Table,
        expected: Table,
    ) -> None:
        actual = table_factory().add_table_as_columns(other)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        other: Table,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.add_table_as_columns(other)
        assert original == table_factory()


def test_should_raise_if_row_counts_differ() -> None:
    with pytest.raises(LengthMismatchError):
        Table({"col1": [1]}).add_table_as_columns(Table({"col2": [1, 2]}))


def test_should_raise_if_duplicate_column_name() -> None:
    with pytest.raises(DuplicateColumnError):
        Table({"col1": [1]}).add_table_as_columns(Table({"col1": [1]}))
