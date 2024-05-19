import polars as pl
import pytest
from safeds.data.tabular.containers import StringCell
from safeds.data.tabular.containers._lazy_string_cell import _LazyStringCell


def test_should_be_deterministic() -> None:
    cell = _LazyStringCell(pl.col("a"))
    assert hash(cell) == 7139977585477665635


@pytest.mark.parametrize(
    ("cell1", "cell2", "expected"),
    [
        (_LazyStringCell(pl.col("a")), _LazyStringCell(pl.col("a")), True),
        (_LazyStringCell(pl.col("a")), _LazyStringCell(pl.col("b")), False),
    ],
    ids=[
        "equal",
        "different",
    ],
)
def test_should_be_good_hash(cell1: StringCell, cell2: StringCell, expected: bool) -> None:
    assert (hash(cell1) == hash(cell2)) == expected
