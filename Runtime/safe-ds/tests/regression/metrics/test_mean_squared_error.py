import pandas as pd
import pytest
from safeds.data import Column
from safeds.regression.metrics import mean_squared_error


@pytest.mark.parametrize(
    "actual, expected, result",
    [([1, 2], [1, 2], 0), ([0, 0], [1, 1], 1), ([1, 1, 1], [2, 2, 11], 34)],
)
def test_mean_squared_error_valid(
    actual: list[float], expected: list[float], result: float
) -> None:
    actual_column: Column = Column(pd.Series(actual), "actual")
    expected_column: Column = Column(pd.Series(expected), "expected")
    assert mean_squared_error(actual_column, expected_column) == result
