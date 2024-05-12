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
        (pl.Series([True, False, True]), [True, False, True]),
        (pl.Series([1, 2, 3]), [1, 2, 3]),
        (pl.Series([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]),
        (pl.Series(["a", "b", "c"]), ["a", "b", "c"]),
    ],
    ids=[
        "empty",
        "boolean",
        "integer",
        "real number",
        "string",
    ],
)
def test_should_store_the_data(series: pl.Series, expected: Column) -> None:
    assert list(Column._from_polars_series(series)) == expected
