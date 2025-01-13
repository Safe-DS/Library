import polars as pl
import pytest

from safeds.data.tabular.containers import Column


def test_should_store_the_name() -> None:
    series = pl.Series("a", [])
    assert Column._from_polars_series(series).name == "a"


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pl.Series([]), []),
        (pl.Series([True]), [True]),
        (pl.Series([1, 2]), [1, 2]),
    ],
    ids=[
        "empty",
        "one row",
        "multiple rows",
    ],
)
def test_should_store_the_data(series: pl.Series, expected: Column) -> None:
    assert list(Column._from_polars_series(series)) == expected
