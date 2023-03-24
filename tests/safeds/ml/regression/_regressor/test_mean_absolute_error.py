import pytest
from ._dummy_regressor import DummyRegressor

from safeds.data.tabular.containers import Column, TaggedTable, Table


@pytest.mark.parametrize(
    "predicted, expected, result",
    [
        ([1, 2], [1, 2], 0),
        ([0, 0], [1, 1], 1),
        ([1, 1, 1], [2, 2, 11], 4),
        ([0, 0, 0], [10, 2, 18], 10),
        ([0.5, 0.5], [1.5, 1.5], 1),
    ],
)
def test_mean_absolute_error_valid(
    predicted: list[float], expected: list[float], result: float
) -> None:
    predicted_column = Column(predicted, "predicted")
    expected_column = Column(expected, "expected")
    table = TaggedTable(Table.from_columns([predicted_column, expected_column]), target_name="expected")

    assert DummyRegressor().mean_absolute_error(table) == result
