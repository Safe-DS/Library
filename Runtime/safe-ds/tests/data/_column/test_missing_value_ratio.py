import numpy as np
import pandas as pd
import pytest
from safeds.data import Column
from safeds.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    "values, expected",
    [([1, 2, 3], 0), ([1, 2, 3, None], 1 / 4), ([None, None, None], 1)],
)
def test_missing_value_ratio(values: list, expected: float) -> None:
    column = Column(pd.Series(values), "A")
    result = column.missing_value_ratio()
    assert result == expected


def test_missing_value_ratio_empty() -> None:
    column = Column(pd.Series([], dtype=np.dtype("float64")), "A")
    with pytest.raises(ColumnSizeError):
        column.missing_value_ratio()
