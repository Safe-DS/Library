from collections.abc import Callable

import polars as pl
import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import DurationOperations


@pytest.mark.parametrize(
    "ops_factory",
    [
        lambda: Cell.duration(hours=1).dur,
        lambda: _LazyCell(pl.col("a")).dur,
    ],
    ids=[
        "duration",
        "column",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, ops_factory: Callable[[], DurationOperations]) -> None:
        ops_1 = ops_factory()
        ops_2 = ops_factory()
        assert hash(ops_1) == hash(ops_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        ops_factory: Callable[[], DurationOperations],
        snapshot: SnapshotAssertion,
    ) -> None:
        ops = ops_factory()
        assert hash(ops) == snapshot


@pytest.mark.parametrize(
    ("ops_1", "ops_2"),
    [
        # different durations
        (
            Cell.duration(hours=1).dur,
            Cell.duration(hours=2).dur,
        ),
        # different columns
        (
            _LazyCell(pl.col("a")).dur,
            _LazyCell(pl.col("b")).dur,
        ),
        # different cell kinds
        (
            Cell.duration(hours=1).dur,
            _LazyCell(pl.col("a")).dur,
        ),
    ],
    ids=[
        "different durations",
        "different columns",
        "different cell kinds",
    ],
)
def test_should_be_good_hash(ops_1: DurationOperations, ops_2: DurationOperations) -> None:
    assert hash(ops_1) != hash(ops_2)
