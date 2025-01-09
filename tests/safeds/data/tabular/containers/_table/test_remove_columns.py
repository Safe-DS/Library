from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table_factory", "names", "ignore_unknown_names", "expected"),
    [
        (
            lambda: Table({}),
            [],
            False,
            Table({}),
        ),
        (
            lambda: Table({}),
            "col1",
            True,
            Table({}),
        ),
        (
            lambda: Table({}),
            ["col1", "col2"],
            True,
            Table({}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            [],
            False,
            Table({"col1": [], "col2": []}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            ["col2"],
            False,
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [], "col2": []}),
            ["col1", "col2"],
            False,
            Table({}),
        ),
    ],
    ids=[
        "empty table, empty list",
        "empty table, single column (ignoring unknown names)",
        "empty table, multiple columns (ignoring unknown names)",
        "non-empty table, empty list",
        "non-empty table, single column",
        "non-empty table, multiple columns",
    ],
)
class TestHappyPath:
    def test_should_remove_columns(
        self,
        table_factory: Callable[[], Table],
        names: str | list[str],
        ignore_unknown_names: bool,
        expected: Table,
    ) -> None:
        actual = table_factory().remove_columns(names, ignore_unknown_names=ignore_unknown_names)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        names: str | list[str],
        ignore_unknown_names: bool,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.remove_columns(names, ignore_unknown_names=ignore_unknown_names)
        assert original == table_factory()


def test_should_raise_for_unknown_columns() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).remove_columns(["col1"])
