import polars as pl
import pytest

from safeds.data.tabular.containers import Column


def test_should_store_the_name() -> None:
    series = pl.Series("col1", [])
    assert Column._from_polars_series(series).name == "col1"


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pl.Series([]), []),
        (pl.Series([True]), [True]),
    ],
    ids=[
        "empty",
        "non-empty",
    ],
)
def test_should_store_the_data(series: pl.Series, expected: Column) -> None:
    assert list(Column._from_polars_series(series)) == expected
