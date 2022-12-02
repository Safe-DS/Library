import numpy as np
import pandas as pd
import pytest
from safe_ds.data import Column, ColumnType
from safe_ds.exceptions import ColumnLengthMismatchError
from safe_ds.regression.metrics import mean_squared_error


@pytest.mark.parametrize(
    "actual, expected, result",
    [([1, 2], [1, 2], 0), ([0, 0], [1, 1], 1), ([1, 1, 1], [2, 2, 11], 34)],
)
def test_mean_squared_error_valid(
    actual: list[float], expected: list[float], result: float
) -> None:
    actual_column: Column = Column(
        pd.Series(actual), "actual", ColumnType.from_numpy_dtype(np.dtype(float))
    )
    expected_column: Column = Column(
        pd.Series(expected), "expected", ColumnType.from_numpy_dtype(np.dtype(float))
    )
    assert mean_squared_error(actual_column, expected_column) == result


@pytest.mark.parametrize(
    "actual, expected, error",
    [
        (["A", "B"], [1, 2], TypeError),
        ([1, 2], ["A", "B"], TypeError),
        ([1, 2, 3], [1, 2], ColumnLengthMismatchError),
    ],
)
def test_mean_squared_error_invalid(
    actual: list[str | int], expected: list[str | int], error: type[Exception]
) -> None:
    actual_column = Column(
        pd.Series(actual),
        "actual",
        ColumnType.from_numpy_dtype(pd.Series(actual).dtype),
    )
    expected_column = Column(
        pd.Series(expected),
        "expected",
        ColumnType.from_numpy_dtype(pd.Series(expected).dtype),
    )
    with pytest.raises(error):
        mean_squared_error(actual_column, expected_column)
