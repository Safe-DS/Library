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
            '_LazyTemporalOperations(1h.alias("duration"))',
        ),
        (
            _LazyCell(pl.col("a")).dt,
            '_LazyTemporalOperations(col("a"))',
        ),
    ],
    ids=[
        "duration",
        "column",
    ],
)
def test_should_return_a_string_representation(ops: TemporalOperations, expected: str) -> None:
    # We do not care about the exact string representation, this is only for debugging
    assert repr(ops) == expected
