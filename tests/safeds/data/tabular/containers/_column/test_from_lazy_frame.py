import polars as pl
import pytest

from safeds.data.tabular.containers import Column


def test_should_store_the_name() -> None:
    frame = pl.LazyFrame({"col1": []})
    assert Column._from_polars_lazy_frame("col1", frame).name == "col1"


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (pl.LazyFrame({"col1": []}), []),
        (pl.LazyFrame({"col1": [True]}), [True]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_store_the_data(frame: pl.LazyFrame, expected: list) -> None:
    assert list(Column._from_polars_lazy_frame("col1", frame)) == expected
