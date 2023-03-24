import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable

from ._dummy_regressor import DummyRegressor


@pytest.mark.parametrize(
    "predicted, expected, result",
    [([1, 2], [1, 2], 0), ([0, 0], [1, 1], 1), ([1, 1, 1], [2, 2, 11], 34)],
)
def test_mean_squared_error_valid(
    predicted: list[float], expected: list[float], result: float
) -> None:
    predicted_column = Column(predicted, "predicted")
    expected_column = Column(expected, "expected")
    table = TaggedTable(
        Table.from_columns([predicted_column, expected_column]), target_name="expected"
    )

    assert DummyRegressor().mean_squared_error(table) == result
