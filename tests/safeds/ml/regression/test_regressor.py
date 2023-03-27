import pandas as pd
import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import ColumnLengthMismatchError
from safeds.ml.regression import Regressor

# noinspection PyProtectedMember
from safeds.ml.regression._regressor import _check_metrics_preconditions


class DummyRegressor(Regressor):
    """
    Dummy regressor to test metrics.

    Metrics methods expect a `TaggedTable` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def fit(self, training_set: TaggedTable) -> None:
        pass

    def predict(self, dataset: Table) -> TaggedTable:
        # Needed until https://github.com/Safe-DS/Stdlib/issues/75 is fixed
        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return TaggedTable(dataset, target_name="predicted")


class TestMeanAbsoluteError:
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
    def test_valid_data(
        self, predicted: list[float], expected: list[float], result: float
    ) -> None:
        predicted_column = Column("predicted", predicted)
        expected_column = Column("expected", expected)
        table = TaggedTable(
            Table.from_columns([predicted_column, expected_column]),
            target_name="expected",
        )

        assert DummyRegressor().mean_absolute_error(table) == result


class TestMeanSquaredError:
    @pytest.mark.parametrize(
        "predicted, expected, result",
        [([1, 2], [1, 2], 0), ([0, 0], [1, 1], 1), ([1, 1, 1], [2, 2, 11], 34)],
    )
    def test_valid_data(
        self, predicted: list[float], expected: list[float], result: float
    ) -> None:
        predicted_column = Column("predicted", predicted)
        expected_column = Column("expected", expected)
        table = TaggedTable(
            Table.from_columns([predicted_column, expected_column]),
            target_name="expected",
        )

        assert DummyRegressor().mean_squared_error(table) == result


class TestCheckMetricsPreconditions:
    @pytest.mark.parametrize(
        "actual, expected, error",
        [
            (["A", "B"], [1, 2], TypeError),
            ([1, 2], ["A", "B"], TypeError),
            ([1, 2, 3], [1, 2], ColumnLengthMismatchError),
        ],
    )
    def test_should_raise_if_validation_fails(
        self, actual: list[str | int], expected: list[str | int], error: type[Exception]
    ) -> None:
        actual_column = Column("actual", pd.Series(actual))
        expected_column = Column("expected", pd.Series(expected))
        with pytest.raises(error):
            _check_metrics_preconditions(actual_column, expected_column)
