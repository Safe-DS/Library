import pytest

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError, PredictionError
from safeds.ml.regression import GradientBoosting, Regressor
from tests.fixtures import resolve_resource_path


@pytest.fixture()
def regressor() -> Regressor:
    return GradientBoosting()


@pytest.fixture()
def valid_data() -> TaggedTable:
    table = Table.from_csv_file(
        resolve_resource_path("test_gradient_boosting_regression.csv")
    )
    return TaggedTable(table, "T")


@pytest.fixture()
def invalid_data() -> TaggedTable:
    table = Table.from_csv_file(
        resolve_resource_path("test_gradient_boosting_regression_invalid.csv")
    )
    return TaggedTable(table, "T")


class TestFit:
    def test_should_succeed_on_valid_data(
        self, regressor: Regressor, valid_data: TaggedTable
    ) -> None:
        regressor.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_raise_on_invalid_data(
        self, regressor: Regressor, invalid_data: TaggedTable
    ) -> None:
        with pytest.raises(LearningError):
            regressor.fit(invalid_data)


class TestPredict:
    def test_should_include_features_of_prediction_input(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_set_correct_target_name(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.target.name == "T"

    def test_should_raise_when_not_fitted(
        self, regressor: Regressor, valid_data: TaggedTable
    ) -> None:
        with pytest.raises(PredictionError):
            regressor.predict(valid_data.features)

    def test_should_raise_on_invalid_data(
        self, regressor: Regressor, valid_data: TaggedTable, invalid_data: TaggedTable
    ) -> None:
        fitted_regressor = regressor.fit(valid_data)
        with pytest.raises(PredictionError):
            fitted_regressor.predict(invalid_data.features)
