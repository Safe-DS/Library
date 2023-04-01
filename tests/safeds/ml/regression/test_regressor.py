from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import ColumnLengthMismatchError, LearningError, PredictionError
from safeds.ml.regression import (
    AdaBoost,
    DecisionTree,
    ElasticNetRegression,
    GradientBoosting,
    KNearestNeighbors,
    LassoRegression,
    LinearRegression,
    RandomForest,
    Regressor,
    RidgeRegression,
)

# noinspection PyProtectedMember
from safeds.ml.regression._regressor import _check_metrics_preconditions

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


def regressors() -> list[Regressor]:
    """
    Return the list of regressors to test.

    After you implemented a new regressor, add it to this list to ensure its `fit` and `predict` method work as
    expected. Place tests of methods that are specific to your regressor in a separate test file.

    Returns
    -------
    regressors : list[Regressor]
        The list of regressors to test.
    """
    return [
        AdaBoost(),
        DecisionTree(),
        ElasticNetRegression(),
        GradientBoosting(),
        KNearestNeighbors(2),
        LassoRegression(),
        LinearRegression(),
        RandomForest(),
        RidgeRegression(),
    ]


@pytest.fixture()
def valid_data() -> TaggedTable:
    return Table.from_columns(
        [
            Column("id", [1, 4]),
            Column("feat1", [2, 5]),
            Column("feat2", [3, 6]),
            Column("target", [0, 1]),
        ],
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


@pytest.fixture()
def invalid_data() -> TaggedTable:
    return Table.from_columns(
        [
            Column("id", [1, 4]),
            Column("feat1", ["a", 5]),
            Column("feat2", [3, 6]),
            Column("target", [0, 1]),
        ],
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


@pytest.mark.parametrize("regressor", regressors(), ids=lambda x: x.__class__.__name__)
class TestFit:
    def test_should_succeed_on_valid_data(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        regressor.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_not_change_input_regressor(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        regressor.fit(valid_data)
        assert not regressor.is_fitted()

    def test_should_not_change_input_table(self, regressor: Regressor, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        regressor.fit(valid_data)
        assert valid_data == valid_data_copy

    def test_should_raise_on_invalid_data(self, regressor: Regressor, invalid_data: TaggedTable) -> None:
        with pytest.raises(LearningError):
            regressor.fit(invalid_data)


@pytest.mark.parametrize("regressor", regressors(), ids=lambda x: x.__class__.__name__)
class TestPredict:
    def test_should_include_features_of_input_table(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_include_complete_input_table(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.remove_columns(["target"]))
        assert prediction.remove_columns(["target"]) == valid_data.remove_columns(["target"])

    def test_should_set_correct_target_name(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.target.name == "target"

    def test_should_not_change_input_table(self, regressor: Regressor, request: FixtureRequest) -> None:
        valid_data = request.getfixturevalue("valid_data")
        valid_data_copy = request.getfixturevalue("valid_data")
        fitted_classifier = regressor.fit(valid_data)
        fitted_classifier.predict(valid_data.features)
        assert valid_data == valid_data_copy

    def test_should_raise_when_not_fitted(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        with pytest.raises(PredictionError):
            regressor.predict(valid_data.features)

    def test_should_raise_on_invalid_data(
        self,
        regressor: Regressor,
        valid_data: TaggedTable,
        invalid_data: TaggedTable,
    ) -> None:
        fitted_regressor = regressor.fit(valid_data)
        with pytest.raises(PredictionError):
            fitted_regressor.predict(invalid_data.features)


@pytest.mark.parametrize("regressor", regressors(), ids=lambda x: x.__class__.__name__)
class TestIsFitted:
    def test_should_return_false_before_fitting(self, regressor: Regressor) -> None:
        assert not regressor.is_fitted()

    def test_should_return_true_after_fitting(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        assert fitted_regressor.is_fitted()


class DummyRegressor(Regressor):
    """
    Dummy regressor to test metrics.

    Metrics methods expect a `TaggedTable` as input with two columns:

    - `predicted`: The predicted targets.
    - `expected`: The correct targets.

    `target_name` must be set to `"expected"`.
    """

    def fit(self, training_set: TaggedTable) -> DummyRegressor:
        # pylint: disable=unused-argument
        return self

    def predict(self, dataset: Table) -> TaggedTable:
        # Needed until https://github.com/Safe-DS/Stdlib/issues/75 is fixed
        predicted = dataset.get_column("predicted")
        feature = predicted.rename("feature")
        dataset = Table.from_columns([feature, predicted])

        return dataset.tag_columns(target_name="predicted")

    def is_fitted(self) -> bool:
        return True


class TestMeanAbsoluteError:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [
            ([1, 2], [1, 2], 0),
            ([0, 0], [1, 1], 1),
            ([1, 1, 1], [2, 2, 11], 4),
            ([0, 0, 0], [10, 2, 18], 10),
            ([0.5, 0.5], [1.5, 1.5], 1),
        ],
    )
    def test_valid_data(self, predicted: list[float], expected: list[float], result: float) -> None:
        predicted_column = Column("predicted", predicted)
        expected_column = Column("expected", expected)
        table = Table.from_columns([predicted_column, expected_column]).tag_columns(
            target_name="expected",
        )

        assert DummyRegressor().mean_absolute_error(table) == result


class TestMeanSquaredError:
    @pytest.mark.parametrize(
        ("predicted", "expected", "result"),
        [([1, 2], [1, 2], 0), ([0, 0], [1, 1], 1), ([1, 1, 1], [2, 2, 11], 34)],
    )
    def test_valid_data(self, predicted: list[float], expected: list[float], result: float) -> None:
        predicted_column = Column("predicted", predicted)
        expected_column = Column("expected", expected)
        table = Table.from_columns([predicted_column, expected_column]).tag_columns(
            target_name="expected",
        )

        assert DummyRegressor().mean_squared_error(table) == result


class TestCheckMetricsPreconditions:
    @pytest.mark.parametrize(
        ("actual", "expected", "error"),
        [
            (["A", "B"], [1, 2], TypeError),
            ([1, 2], ["A", "B"], TypeError),
            ([1, 2, 3], [1, 2], ColumnLengthMismatchError),
        ],
    )
    def test_should_raise_if_validation_fails(
        self,
        actual: list[str | int],
        expected: list[str | int],
        error: type[Exception],
    ) -> None:
        actual_column = Column("actual", pd.Series(actual))
        expected_column = Column("expected", pd.Series(expected))
        with pytest.raises(error):
            _check_metrics_preconditions(actual_column, expected_column)
