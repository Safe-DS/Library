from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.query import ColumnSelector
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
            lambda: Table({"col1": [], "col2": []}),
            [],
            Table({"col1": [], "col2": []}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            ["col2"],
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            ["col1", "col2"],
            Table({}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            ColumnSelector.by_name("col1"),
            Table({"col2": []}),
        ),
    ],
    ids=[
        "empty table, empty list",
        "non-empty table, empty list",
        "non-empty table, single column",
        "non-empty table, multiple columns",
        "selector",
    ],
)
class TestHappyPath:
    def test_should_remove_columns(
        self,
        table_factory: Callable[[], Table],
        names: str | list[str],
        expected: Table,
    ) -> None:
        actual = table_factory().remove_columns(names)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        names: str | list[str],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.remove_columns(names)
        assert original == table_factory()


def test_should_raise_for_unknown_columns() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).remove_columns(["col1"])
