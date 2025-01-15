from collections.abc import Callable

import polars as pl
import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import TemporalOperations


@pytest.mark.parametrize(
    "ops_factory",
    [
        lambda: Cell.duration(hours=1).dt,
        lambda: _LazyCell(pl.col("a")).dt,
    ],
    ids=[
        "duration",
        "column",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, ops_factory: Callable[[], TemporalOperations]) -> None:
        ops_1 = ops_factory()
        ops_2 = ops_factory()
        assert hash(ops_1) == hash(ops_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        ops_factory: Callable[[], TemporalOperations],
        snapshot: SnapshotAssertion,
    ) -> None:
        ops = ops_factory()
        assert hash(ops) == snapshot


@pytest.mark.parametrize(
    ("ops_1", "ops_2"),
    [
        # different durations
        (
            Cell.duration(hours=1).dt,
            Cell.duration(hours=2).dt,
        ),
        # different column
        (
            _LazyCell(pl.col("a")).dt,
            _LazyCell(pl.col("b")).dt,
        ),
        # different cell kinds
        (
            Cell.duration(hours=1).dt,
            _LazyCell(pl.col("a")).dt,
        ),
    ],
    ids=[
        "different durations",
        "different column",
        "different cell kinds",
    ],
)
def test_should_be_good_hash(ops_1: TemporalOperations, ops_2: TemporalOperations) -> None:
    assert hash(ops_1) != hash(ops_2)
