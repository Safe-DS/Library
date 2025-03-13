from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.query import ColumnSelector


@pytest.mark.parametrize(
    ("table_factory", "column_names", "expected"),
    [
        # empty
        (
            lambda: Table({}),
            None,
            Table({}),
        ),
        # no rows
        (
            lambda: Table({"col1": []}),
            None,
            Table({"col1": []}),
        ),
        # no missing values
        (
            lambda: Table({"col1": [1, 2]}),
            None,
            Table({"col1": [1, 2]}),
        ),
        # missing values (all columns selected)
        (
            lambda: Table({"col1": [1, 2, None], "col2": [1, None, 3], "col3": [None, 2, 3]}),
            None,
            Table({"col1": [], "col2": [], "col3": []}),
        ),
        # missing values (several columns selected)
        (
            lambda: Table({"col1": [1, 2, None], "col2": [1, None, 3], "col3": [None, 2, 3]}),
            ["col1", "col2"],
            Table({"col1": [1], "col2": [1], "col3": [None]}),
        ),
        # missing values (one column selected)
        (
            lambda: Table({"col1": [1, 2, None], "col2": [1, None, 3], "col3": [None, 2, 3]}),
            "col1",
            Table({"col1": [1, 2], "col2": [1, None], "col3": [None, 2]}),
        ),
        # missing values (no columns selected)
        (
            lambda: Table({"col1": [1, 2, None], "col2": [1, None, 3], "col3": [None, 2, 3]}),
            [],
            Table({"col1": [1, 2, None], "col2": [1, None, 3], "col3": [None, 2, 3]}),
        ),
        # selector
        (
            lambda: Table({"col1": [1, 2, None], "col2": [1, None, 3], "col3": [None, 2, 3]}),
            ColumnSelector.by_name("col1"),
            Table({"col1": [1, 2], "col2": [1, None], "col3": [None, 2]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "no missing values",
        "missing values (all columns selected)",
        "missing values (several columns selected)",
        "missing values (one column selected)",
        "missing values (no columns selected)",
        "selector",
    ],
)
class TestHappyPath:
    def test_should_remove_rows_with_missing_values(
        self,
        table_factory: Callable[[], Table],
        column_names: str | list[str] | None,
        expected: Table,
    ) -> None:
        actual = table_factory().remove_rows_with_missing_values(selector=column_names)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        column_names: str | list[str] | None,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.remove_rows_with_missing_values(selector=column_names)
        assert original == table_factory()
