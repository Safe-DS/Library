import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (
            Cell.constant(1),
            "_LazyCell(dyn int: 1)",
        ),
        (
            _LazyCell(pl.col("a")),
            '_LazyCell(col("a"))',
        ),
    ],
    ids=[
        "constant",
        "column",
    ],
)
def test_should_return_a_string_representation(cell: Cell, expected: str) -> None:
    # We do not care about the exact string representation, this is only for debugging
    assert repr(cell) == expected
