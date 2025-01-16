import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import MathOperations


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        (
            Cell.constant(1).math,
            "_LazyMathOperations(dyn int: 1)",
        ),
        (
            _LazyCell(pl.col("a")).math,
            '_LazyMathOperations(col("a"))',
        ),
    ],
    ids=[
        "constant",
        "column",
    ],
)
def test_should_return_a_string_representation(ops: MathOperations, expected: str) -> None:
    # We do not care about the exact string representation, this is only for debugging
    assert repr(ops) == expected
