import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import TemporalOperations


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        (
            Cell.duration(hours=1).dt,
            '(1h.alias("duration")).dt',
        ),
        (
            _LazyCell(pl.col("a")).dt,
            '(col("a")).dt',
        ),
    ],
    ids=[
        "constant",
        "column",
    ],
)
def test_should_return_a_string_representation(ops: TemporalOperations, expected: str) -> None:
    # We do not care about the exact string representation, this is only for debugging
    assert str(ops) == expected
