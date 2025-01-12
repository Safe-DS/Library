from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table_factory", "name", "descending", "expected"),
    [
        (
            lambda: Table({"col1": [], "col2": []}),
            "col1",
            False,
            Table({"col1": [], "col2": []}),
        ),
        (
            lambda: Table({"col1": [2, 3, 1], "col2": [5, 6, 4]}),
            "col1",
            False,
            Table({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
        ),
        (
            lambda: Table({"col1": [2, 3, 1], "col2": [5, 6, 4]}),
            "col1",
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
        name: str,
        descending: bool,
        expected: Table,
    ) -> None:
        actual = table_factory().sort_rows_by_column(name, descending=descending)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        name: str,
        descending: bool,
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.sort_rows_by_column(name, descending=descending)
        assert original == table_factory()


def test_should_raise_if_name_is_unknown() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).sort_rows_by_column("col1")
