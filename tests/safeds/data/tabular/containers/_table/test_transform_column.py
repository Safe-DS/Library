from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Cell, Table
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table_factory", "name", "transformer", "expected"),
    [
        (
            lambda: Table({"col1": []}),
            "col1",
            lambda _: Cell.from_literal(None),
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": []}),
            "col1",
            lambda cell: 2 * cell,
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            "col1",
            lambda _: Cell.from_literal(None),
            Table({"col1": [None, None, None]}),
        ),
        (
            lambda: Table({"col1": [1, 2, 3]}),
            "col1",
            lambda cell: 2 * cell,
            Table({"col1": [2, 4, 6]}),
        ),
    ],
    ids=[
        "no rows (constant value)",
        "no rows (computed value)",
        "non-empty (constant value)",
        "non-empty (computed value)",
    ],
)
class TestHappyPath:
    def test_should_transform_column(
        self,
        table_factory: Callable[[], Table],
        name: str,
        transformer: Callable[[Cell], Cell],
        expected: Table,
    ) -> None:
        actual = table_factory().transform_column(name, transformer)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        name: str,
        transformer: Callable[[Cell], Cell],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.transform_column(name, transformer)
        assert original == table_factory()


def test_should_raise_if_column_not_found() -> None:
    with pytest.raises(ColumnNotFoundError):
        Table({}).transform_column("col1", lambda cell: cell * 2)
