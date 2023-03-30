import pytest
from safeds.data.tabular.containers import Column, Table, TaggedTable
from safeds.exceptions import LearningError, PredictionError
from safeds.ml.regression import ElasticNetRegression, Regressor


@pytest.fixture()
def regressor() -> Regressor:
    return ElasticNetRegression()


@pytest.fixture()
def valid_data() -> TaggedTable:
    return Table.from_columns(
        [
            Column("id", [1, 4]),
            Column("feat1", [2, 5]),
            Column("feat2", [3, 6]),
            Column("target", [0, 1]),
        ]
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


@pytest.fixture()
def invalid_data() -> TaggedTable:
    return Table.from_columns(
        [
            Column("id", [1, 4]),
            Column("feat1", ["a", 5]),
            Column("feat2", [3, 6]),
            Column("target", [0, 1]),
        ]
    ).tag_columns(target_name="target", feature_names=["feat1", "feat2"])


class TestFit:
    def test_should_succeed_on_valid_data(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        regressor.fit(valid_data)
        assert True  # This asserts that the fit method succeeds

    def test_should_raise_on_invalid_data(self, regressor: Regressor, invalid_data: TaggedTable) -> None:
        with pytest.raises(LearningError):
            regressor.fit(invalid_data)


class TestPredict:
    def test_should_include_features_of_prediction_input(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.features == valid_data.features

    def test_should_include_complete_prediction_input(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.remove_columns(["target"]))
        assert prediction.remove_columns(["target"]) == valid_data.remove_columns(["target"])

    def test_should_set_correct_target_name(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        fitted_regressor = regressor.fit(valid_data)
        prediction = fitted_regressor.predict(valid_data.features)
        assert prediction.target.name == "target"

    def test_should_raise_when_not_fitted(self, regressor: Regressor, valid_data: TaggedTable) -> None:
        with pytest.raises(PredictionError):
            regressor.predict(valid_data.features)

    def test_should_raise_on_invalid_data(
        self, regressor: Regressor, valid_data: TaggedTable, invalid_data: TaggedTable
    ) -> None:
        fitted_regressor = regressor.fit(valid_data)
        with pytest.raises(PredictionError):
            fitted_regressor.predict(invalid_data.features)
