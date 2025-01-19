from collections.abc import Callable

import polars as pl
import pytest
from syrupy import SnapshotAssertion

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import StringOperations


@pytest.mark.parametrize(
    "ops_factory",
    [
        lambda: Cell.constant("a").str,
        lambda: _LazyCell(pl.col("a")).str,
    ],
    ids=[
        "constant",
        "column",
    ],
)
class TestContract:
    def test_should_return_same_hash_for_equal_objects(self, ops_factory: Callable[[], StringOperations]) -> None:
        ops_1 = ops_factory()
        ops_2 = ops_factory()
        assert hash(ops_1) == hash(ops_2)

    def test_should_return_same_hash_in_different_processes(
        self,
        ops_factory: Callable[[], StringOperations],
        snapshot: SnapshotAssertion,
    ) -> None:
        ops = ops_factory()
        assert hash(ops) == snapshot


@pytest.mark.parametrize(
    ("ops_1", "ops_2"),
    [
        # different constant values
        (
            Cell.constant("a").str,
            Cell.constant("b").str,
        ),
        # different columns
        (
            _LazyCell(pl.col("a")).str,
            _LazyCell(pl.col("b")).str,
        ),
        # different cell kinds
        (
            Cell.constant("a").str,
            _LazyCell(pl.col("a")).str,
        ),
    ],
    ids=[
        "different constant values",
        "different columns",
        "different cell kinds",
    ],
)
def test_should_be_good_hash(ops_1: StringOperations, ops_2: StringOperations) -> None:
    assert hash(ops_1) != hash(ops_2)
