from collections.abc import Callable

import polars as pl
import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import MathOperations


@pytest.mark.parametrize(
    "ops_factory",
    [
        lambda: Cell.constant(1).math,
        lambda: _LazyCell(pl.col("a")).math,
    ],
    ids=[
        "constant",
        "column",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, ops_factory: Callable[[], MathOperations]) -> None:
        ops_1 = ops_factory()
        ops_2 = ops_factory()
        assert hash(ops_1) == hash(ops_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        ops_factory: Callable[[], MathOperations],
        snapshot: SnapshotAssertion,
    ) -> None:
        ops = ops_factory()
        assert hash(ops) == snapshot


@pytest.mark.parametrize(
    ("ops_1", "ops_2"),
    [
        # different constant values
        (
            Cell.constant(1).math,
            Cell.constant(2).math,
        ),
        # different columns
        (
            _LazyCell(pl.col("a")).math,
            _LazyCell(pl.col("b")).math,
        ),
        # different cell kinds
        (
            Cell.constant(1).math,
            _LazyCell(pl.col("a")).math,
        ),
    ],
    ids=[
        "different constant values",
        "different columns",
        "different cell kinds",
    ],
)
def test_should_be_good_hash(ops_1: MathOperations, ops_2: MathOperations) -> None:
    assert hash(ops_1) != hash(ops_2)
