import polars as pl
import pytest
from safeds.data.tabular.containers import TemporalCell
from safeds.data.tabular.containers._lazy_temporal_cell import _LazyTemporalCell


def test_should_be_deterministic() -> None:
    cell = _LazyTemporalCell(pl.col("a"))
    assert hash(cell) == 7139977585477665635


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyTemporalCell(pl.col("a")), _LazyTemporalCell(pl.col("a")), True),
        (_LazyTemporalCell(pl.col("a")), _LazyTemporalCell(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_be_good_hash(cell1: TemporalCell, cell2: TemporalCell, expected: bool) -> None:
    assert (hash(cell1) == hash(cell2)) == expected
