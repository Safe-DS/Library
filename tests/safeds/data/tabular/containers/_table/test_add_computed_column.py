from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Cell, Row, Table
from safeds.exceptions import DuplicateColumnError


@pytest.mark.parametrize(
    ("table_factory", "name", "computer", "expected"),
    [
        (
            lambda: Table({}),
            "col1",
            lambda _: Cell.from_literal(None),
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": []}),
            "col2",
            lambda _: Cell.from_literal(None),
            Table({"col1": [], "col2": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            "col2",
            lambda _: Cell.from_literal(None),
            Table({"col1": [1, 2, 3], "col2": [None, None, None]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            "col2",
            lambda row: 2 * row["col1"],
            Table({"col1": [1, 2, 3], "col2": [2, 4, 6]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "non-empty, constant value",
        "non-empty, computed value",
    ],
)
class TestHappyPath:
    def test_should_add_computed_column(
        self,
        table_factory: Callable[[], Table],
        name: str,
        computer: Callable[[Row], Cell],
        expected: Table,
    ) -> None:
        actual = table_factory().add_computed_column(name, computer)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        name: str,
        computer: Callable[[Row], Cell],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.add_computed_column(name, computer)
        assert original == table_factory()


def test_should_raise_if_duplicate_column_name() -> None:
    with pytest.raises(DuplicateColumnError):
        Table({"col1": [1]}).add_computed_column("col1", lambda row: row["col1"])
