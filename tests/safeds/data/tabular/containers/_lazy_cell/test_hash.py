from collections.abc import Callable

import polars as pl
import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell


@pytest.mark.parametrize(
    "cell_factory",
    [
        lambda: Cell.constant(1),
        lambda: Cell.date(2025, 1, 15),
        lambda: Cell.date(_LazyCell(pl.col("a")), 1, 15),
        lambda: _LazyCell(pl.col("a")),
    ],
    ids=[
        "constant",
        "date, int",
        "date, column",
        "column",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, cell_factory: Callable[[], Cell]) -> None:
        cell_1 = cell_factory()
        cell_2 = cell_factory()
        assert hash(cell_1) == hash(cell_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        cell_factory: Callable[[], Cell],
        snapshot: SnapshotAssertion,
    ) -> None:
        cell = cell_factory()
        assert hash(cell) == snapshot


@pytest.mark.parametrize(
    ("cell_1", "cell_2"),
    [
        # different constant values
        (
            Cell.constant(1),
            Cell.constant(2),
        ),
        # different constant types
        (
            Cell.constant(1),
            Cell.constant("1"),
        ),
        # different dates, int
        (
            Cell.date(2025, 1, 15),
            Cell.date(2024, 1, 15),
        ),
        # different dates, column
        (
            Cell.date(_LazyCell(pl.col("a")), 1, 15),
            Cell.date(_LazyCell(pl.col("b")), 1, 15),
        ),
        # different columns
        (
            _LazyCell(pl.col("a")),
            _LazyCell(pl.col("b")),
        ),
        # different cell kinds
        (
            Cell.date(23, 1, 15),
            Cell.time(23, 1, 15),
        ),
    ],
    ids=[
        "different constant values",
        "different constant types",
        "different dates, int",
        "different dates, column",
        "different columns",
        "different cell kinds",
    ],
)
def test_should_be_good_hash(cell_1: Cell, cell_2: Cell) -> None:
    assert hash(cell_1) != hash(cell_2)
