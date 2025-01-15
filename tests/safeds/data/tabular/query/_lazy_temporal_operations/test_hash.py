import polars as pl
import pytest

from safeds.data.tabular.query import TemporalOperations
from safeds.data.tabular.query._lazy_temporal_operations import _LazyTemporalOperations


def test_should_be_deterministic() -> None:
    cell = _LazyTemporalOperations(pl.col("a"))
    assert hash(cell) == 8162512882156938440


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyTemporalOperations(pl.col("a")), _LazyTemporalOperations(pl.col("a")), True),
        (_LazyTemporalOperations(pl.col("a")), _LazyTemporalOperations(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_be_good_hash(cell1: TemporalOperations, cell2: TemporalOperations, expected: bool) -> None:
    assert (hash(cell1) == hash(cell2)) == expected
