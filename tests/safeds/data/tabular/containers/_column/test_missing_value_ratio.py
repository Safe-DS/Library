import numpy as np
import pandas as pd
import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 2, 3], 0), ([1, 2, 3, None], 1 / 4), ([None, None, None], 1)],
)
def test_missing_value_ratio(values: list, expected: float) -> None:
    column = Column("A", pd.Series(values))
    result = column.missing_value_ratio()
    assert result == expected


def test_missing_value_ratio_empty() -> None:
    column = Column("A", pd.Series([], dtype=np.dtype("float64")))
    with pytest.raises(ColumnSizeError):
        column.missing_value_ratio()
