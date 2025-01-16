import polars as pl
import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.query import DurationOperations


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        (
            Cell.duration(hours=1).dur,
            '_LazyDurationOperations(1h.alias("duration"))',
        ),
        (
            _LazyCell(pl.col("a")).dur,
            '_LazyDurationOperations(col("a"))',
        ),
    ],
    ids=[
        "duration",
        "column",
    ],
)
def test_should_return_a_string_representation(ops: DurationOperations, expected: str) -> None:
    # We do not care about the exact string representation, this is only for debugging
    assert repr(ops) == expected
