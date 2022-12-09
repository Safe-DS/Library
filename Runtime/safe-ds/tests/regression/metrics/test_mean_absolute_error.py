import numpy as np
import pandas as pd
import pytest
from safe_ds.data import Column, ColumnType
from safe_ds.regression.metrics import mean_absolute_error


@pytest.mark.parametrize(
    "actual, expected, result",
    [
        ([1, 2], [1, 2], 0),
        ([0, 0], [1, 1], 1),
        ([1, 1, 1], [2, 2, 11], 4),
        ([0, 0, 0], [10, 2, 18], 10),
        ([0.5, 0.5], [1.5, 1.5], 1),
    ],
)
def test_mean_absolute_error_valid(
    actual: list[float], expected: list[float], result: float
) -> None:
    actual_column: Column = Column(
        pd.Series(actual), "actual", ColumnType.from_numpy_dtype(np.dtype(float))
    )
    expected_column: Column = Column(
        pd.Series(expected), "expected", ColumnType.from_numpy_dtype(np.dtype(float))
    )
    assert mean_absolute_error(actual_column, expected_column) == result
