from collections.abc import Callable

import pytest

from safeds.data.tabular.containers import Cell, Row, Table
from safeds.data.tabular.query import ColumnSelector
from safeds.exceptions import ColumnNotFoundError


@pytest.mark.parametrize(
    ("table_factory", "selector", "transformer", "expected"),
    [
        # no rows (constant value)
        (
            lambda: Table({"col1": []}),
            "col1",
            lambda _: Cell.constant(None),
            Table({"col1": []}),
        ),
        # no rows (computed value)
        (
            lambda: Table({"col1": []}),
            "col1",
            lambda cell: 2 * cell,
            Table({"col1": []}),
        ),
        # non-empty (constant value)
        (
            lambda: Table({"col1": [1, 2]}),
            "col1",
            lambda _: Cell.constant(None),
            Table({"col1": [None, None]}),
        ),
        # non-empty (computed value)
        (
            lambda: Table({"col1": [1, 2]}),
            "col1",
            lambda cell: 2 * cell,
            Table({"col1": [2, 4]}),
        ),
        # multiple columns transformed (constant value)
        (
            lambda: Table({"col1": [1, 2], "col2": [3, 4]}),
            ["col1", "col2"],
            lambda _: Cell.constant(None),
            Table({"col1": [None, None], "col2": [None, None]}),
        ),
        # multiple columns transformed (computed value)
        (
            lambda: Table({"col1": [1, 2], "col2": [3, 4]}),
            ["col1", "col2"],
            lambda cell: 2 * cell,
            Table({"col1": [2, 4], "col2": [6, 8]}),
        ),
        # lambda takes row parameter
        (
            lambda: Table({"col1": [1, 2], "col2": [3, 4]}),
            "col1",
            lambda cell, row: 2 * cell + row["col2"],
            Table({"col1": [5, 8], "col2": [3, 4]}),
        ),
        # selector
        (
            lambda: Table({"col1": [1, 2], "col2": [3, 4]}),
            ColumnSelector.by_name("col1"),
            lambda _: Cell.constant(None),
            Table({"col1": [None, None], "col2": [3, 4]}),
        )
    ],
    ids=[
        "no rows (constant value)",
        "no rows (computed value)",
        "non-empty (constant value)",
        "non-empty (computed value)",
        "multiple columns transformed (constant value)",
        "multiple columns transformed (computed value)",
        "lambda takes row parameter",
        "selector"
    ],
)
class TestHappyPath:
    def test_should_transform_columns(
        self,
        table_factory: Callable[[], Table],
        selector: str,
        transformer: Callable[[Cell], Cell] | Callable[[Cell, Row], Cell],
        expected: Table,
    ) -> None:
        actual = table_factory().transform_columns(selector, transformer)
        assert actual == expected

    def test_should_not_mutate_receiver(
        self,
        table_factory: Callable[[], Table],
        selector: str,
        transformer: Callable[[Cell], Cell] | Callable[[Cell, Row], Cell],
        expected: Table,  # noqa: ARG002
    ) -> None:
        original = table_factory()
        original.transform_columns(selector, transformer)
        assert original == table_factory()


@pytest.mark.parametrize(
    ("table", "selector"),
    [
        (Table({"col1": [1, 2]}), "col2"),
        (Table({"col1": [1, 2]}), ["col1", "col2"]),
    ],
    ids=[
        "one column name",
        "multiple column names",
    ],
)
def test_should_raise_if_column_not_found(
    table: Table,
    selector: str,
) -> None:
    with pytest.raises(ColumnNotFoundError):
        table.transform_columns(selector, lambda cell: cell * 2)
