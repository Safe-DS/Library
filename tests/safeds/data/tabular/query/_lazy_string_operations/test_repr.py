import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import StringOperations


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        (
            Cell.constant("a").str,
            '_LazyStringOperations("a")',
        ),
        (
            _LazyCell(pl.col("a")).str,
            '_LazyStringOperations(col("a"))',
        ),
    ],
    ids=[
        "constant",
        "column",
    ],
)
def test_should_return_a_string_representation(ops: StringOperations, expected: str) -> None:
    # We do not care about the exact string representation, this is only for debugging
    assert repr(ops) == expected
