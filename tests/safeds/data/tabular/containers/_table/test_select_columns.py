from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table_factory", "names", "expected"),
    [
        (
            lambda: Table({}),
            [],
            Table({}),
        ),
        (
            lambda: Table({}),
            lambda column: column.name.endswith("1"),
            Table({}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            [],
            Table({}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            "col2",
            Table({"col2": []}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            ["col1", "col2"],
            Table({"col1": [], "col2": []}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            lambda column: column.name.endswith("1"),
            Table({"col1": []}),
        ),
        # Related to https://github.com/Safe-DS/Library/issues/115
        (
            lambda: Table({"A": [1], "B": [2], "C": [3]}),
            ["C", "A"],
            Table({"C": [3], "A": [1]}),
        ),
    ],
    ids=[
        "empty table, empty list",
        "empty table, predicate",
        "non-empty table, empty list",
        "non-empty table, single column",
        "non-empty table, multiple columns",
        "non-empty table, predicate",
        "swapped order",
    ],
)
class TestHappyPath:
    def test_should_select_columns(
        self,
        table_factory: Callable[[], Table],
        names: str | list[str],
        expected: Table,
    ) -> None:
        actual = table_factory().select_columns(names)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        names: str | list[str],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.select_columns(names)
        assert original == table_factory()


def test_should_raise_for_unknown_columns() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).select_columns(["col1"])
