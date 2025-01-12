from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import DuplicateColumnError, OutOfBoundsError


@pytest.mark.parametrize(
    ("table_factory", "name", "first_index", "expected"),
    [
        (
            lambda: Table({}),
            "id",
            0,
            Table({"id": []}),
        ),
        (
            lambda: Table({"col1": []}),
            "id",
            0,
            Table({"id": [], "col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            "id",
            0,
            Table({"id": [0, 1, 2], "col1": [1, 2, 3]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            "id",
            10,
            Table({"id": [10, 11, 12], "col1": [1, 2, 3]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "non-empty (first index 0)",
        "non-empty (first index 10)",
    ],
)
class TestHappyPath:
    def test_should_add_index_column(
        self,
        table_factory: Callable[[], Table],
        name: str,
        first_index: int,
        expected: Table,
    ) -> None:
        actual = table_factory().add_index_column(name, first_index=first_index)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        name: str,
        first_index: int,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.add_index_column(name, first_index=first_index)
        assert original == table_factory()


def test_should_raise_if_duplicate_column_name() -> None:
    with pytest.raises(DuplicateColumnError):
        Table({"col1": []}).add_index_column("col1")


def test_should_raise_if_first_index_is_negative() -> None:
    with pytest.raises(OutOfBoundsError):
        Table({}).add_index_column("col1", first_index=-1)
