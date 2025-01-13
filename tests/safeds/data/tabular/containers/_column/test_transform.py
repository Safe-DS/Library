from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Cell, Column


@pytest.mark.parametrize(
    ("column_factory", "transformer", "expected"),
    [
        (
            lambda: Column("col1", []),
            lambda _: Cell.from_literal(None),
            Column("col1", []),
        ),
        (
            lambda: Column("col1", []),
            lambda cell: 2 * cell,
            Column("col1", []),
        ),
        (
            lambda: Column("col1", [1, 2, 3]),
            lambda _: Cell.from_literal(None),
            Column("col1", [None, None, None]),
        ),
        (
            lambda: Column("col1", [1, 2, 3]),
            lambda cell: 2 * cell,
            Column("col1", [2, 4, 6]),
        ),
    ],
    ids=[
        "empty (constant value)",
        "empty (computed value)",
        "non-empty (constant value)",
        "non-empty (computed value)",
    ],
)
class TestHappyPath:
    def test_should_transform_column(
        self,
        column_factory: Callable[[], Column],
        transformer: Callable[[Cell], Cell],
        expected: Column,
    ) -> None:
        actual = column_factory().transform(transformer)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        column_factory: Callable[[], Column],
        transformer: Callable[[Cell], Cell],
        expected: Column,  # noqa: ARG002
    ) -> None:
        original = column_factory()
        original.transform(transformer)
        assert original == column_factory()
