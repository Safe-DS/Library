import typing

import numpy as np
import pandas as pd
import pytest
from safeds.data import Column
from safeds.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3, 4, None], 1 / 4),
        ([1, 1, 3, "abc", None], 2 / 4),
        (["b", "a", "abc", "abc", "abc"], 3 / 5),
    ],
)
def test_stability(values: list[typing.Any], expected: float) -> None:
    column = Column(pd.Series(values), "A")
    assert column.statistics.stability() == expected


def test_stability_error() -> None:
    column = Column(
        pd.Series([], dtype=np.dtype("float64")), "A"
    )  # Fix warning against unknown type
    with pytest.raises(ColumnSizeError):
        column.statistics.stability()
