from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Cell, Row, Table


@pytest.mark.parametrize(
    ("table_factory", "predicate", "expected"),
    [
        (
            lambda: Table({}),
            lambda _: Cell.from_literal(False),  # noqa: FBT003
            Table({}),
        ),
        (
            lambda: Table({"col1": []}),
            lambda _: Cell.from_literal(False),  # noqa: FBT003
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2]}),
            lambda row: row["col1"] <= 0,
            Table({"col1": []}),
        ),
        (
            lambda: Table({"col1": [1, 2]}),
            lambda row: row["col1"] <= 1,
            Table({"col1": [1]}),
        ),
        (
            lambda: Table({"col1": [1, 2]}),
            lambda row: row["col1"] <= 2,
            Table({"col1": [1, 2]}),
        ),
    ],
    ids=[
        "empty",
        "no rows",
        "no matches",
        "some matches",
        "only matches",
    ],
)
class TestHappyPath:
    def test_should_filter_rows(
        self,
        table_factory: Callable[[], Table],
        predicate: Callable[[Row], Cell[bool]],
        expected: Table,
    ) -> None:
        actual = table_factory().filter_rows(predicate)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        predicate: Callable[[Row], Cell[bool]],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.filter_rows(predicate)
        assert original == table_factory()
