import polars as pl
import pytest

from safeds.data.tabular.query import StringOperations
from safeds.data.tabular.query._lazy_string_operations import _LazyStringOperations


def test_should_be_deterministic() -> None:
    cell = _LazyStringOperations(pl.col("a"))
    assert hash(cell) == 8162512882156938440


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyStringOperations(pl.col("a")), _LazyStringOperations(pl.col("a")), True),
        (_LazyStringOperations(pl.col("a")), _LazyStringOperations(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_be_good_hash(cell1: StringOperations, cell2: StringOperations, expected: bool) -> None:
    assert (hash(cell1) == hash(cell2)) == expected
