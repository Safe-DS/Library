from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Cell, Row, Table


@pytest.mark.parametrize(
    ("table_factory", "key_selector", "descending", "expected"),
    [
        (
            lambda: Table({"col1": [], "col2": []}),
            lambda row: row["col1"],
            False,
            Table({"col1": [], "col2": []}),
        ),
        (
            lambda: Table({"col1": [2, 3, 1], "col2": [5, 6, 4]}),
            lambda row: row["col1"],
            False,
            Table({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
        ),
        (
            lambda: Table({"col1": [2, 3, 1], "col2": [5, 6, 4]}),
            lambda row: row["col1"],
            True,
            Table({"col1": [3, 2, 1], "col2": [6, 5, 4]}),
        ),
    ],
    ids=[
        "no rows",
        "non-empty, ascending",
        "non-empty, descending",
    ],
)
class TestHappyPath:
    def test_should_return_sorted_table(
        self,
        table_factory: Callable[[], Table],
        key_selector: Callable[[Row], Cell],
        descending: bool,
        expected: Table,
    ) -> None:
        actual = table_factory().sort_rows(key_selector, descending=descending)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        key_selector: Callable[[Row], Cell],
        descending: bool,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.sort_rows(key_selector, descending=descending)
        assert original == table_factory()
