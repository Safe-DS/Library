from collections.abc import Callable

import polars as pl
import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import DatetimeOperations


@pytest.mark.parametrize(
    "ops_factory",
    [
        lambda: Cell.time(1, 0, 0).dt,
        lambda: _LazyCell(pl.col("a")).dt,
    ],
    ids=[
        "time",
        "column",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, ops_factory: Callable[[], DatetimeOperations]) -> None:
        ops_1 = ops_factory()
        ops_2 = ops_factory()
        assert hash(ops_1) == hash(ops_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        ops_factory: Callable[[], DatetimeOperations],
        snapshot: SnapshotAssertion,
    ) -> None:
        ops = ops_factory()
        assert hash(ops) == snapshot


@pytest.mark.parametrize(
    ("ops_1", "ops_2"),
    [
        # different times
        (
            Cell.time(1, 0, 0).dt,
            Cell.time(2, 0, 0).dt,
        ),
        # different columns
        (
            _LazyCell(pl.col("a")).dt,
            _LazyCell(pl.col("b")).dt,
        ),
        # different cell kinds
        (
            Cell.time(1, 0, 0).dt,
            _LazyCell(pl.col("a")).dt,
        ),
    ],
    ids=[
        "different times",
        "different columns",
        "different cell kinds",
    ],
)
def test_should_be_good_hash(ops_1: DatetimeOperations, ops_2: DatetimeOperations) -> None:
    assert hash(ops_1) != hash(ops_2)
