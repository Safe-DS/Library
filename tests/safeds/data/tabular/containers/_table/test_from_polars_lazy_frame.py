from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"col1": []},
        {"col1": [1, 2], "col2": [3, 4]},
    ],
    ids=[
        "empty",
        "no rows",
        "non-empty",
    ],
)
def test_should_create_table(data: dict[str, list[Any]]) -> None:
    actual = Table._from_polars_lazy_frame(pl.LazyFrame(data))
    expected = Table(data)
    assert actual == expected
