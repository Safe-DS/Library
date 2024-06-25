import pytest
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    ColumnTypeError,
    DatasetMissesDataError,
    FeatureDataMismatchError,
    ModelNotFittedError,
    TargetDataMismatchError,
)
from safeds.ml.classical.regression import BaselineRegressor


class TestBaselineRegressor:
    def test_should_raise_if_fit_dataset_contains_no_data(self) -> None:
        model = BaselineRegressor()
        data = Table({"feat": [], "target": []}).to_tabular_dataset("target")
        with pytest.raises(DatasetMissesDataError):
            model.fit(data)

    def test_should_raise_if_predict_dataset_contains_no_data(self) -> None:
        model = BaselineRegressor()
        fit_data = Table({"feat": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        predict_data = Table({"feat": [], "target": []}).to_tabular_dataset("target")
        model = model.fit(fit_data)
        with pytest.raises(DatasetMissesDataError):
            model.predict(predict_data)

    def test_should_raise_if_fit_dataset_contains_non_numerical_columns(self) -> None:
        model = BaselineRegressor()
        data = Table({"feat": ["a", "b"], "target": [0, 1]}).to_tabular_dataset("target")
        with pytest.raises(ColumnTypeError):
            model.fit(data)

    def test_should_raise_if_predict_dataset_contains_non_numerical_columns(self) -> None:
        model = BaselineRegressor()
        fit_data = Table({"feat": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        predict_data = Table({"feat": ["zero", "one"], "target": [0, 1]}).to_tabular_dataset("target")
        model = model.fit(fit_data)
        with pytest.raises(ColumnTypeError):
            model.predict(predict_data)

    def test_should_check_that_fit_returns_baseline_classifier(self) -> None:
        model = BaselineRegressor()
        data = Table({"feat": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        assert isinstance(model.fit(data), BaselineRegressor)

    def test_should_raise_if_is_fitted_is_set_correctly(self) -> None:
        model = BaselineRegressor()
        data = Table({"feat": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        assert not model.is_fitted
        model = model.fit(data)
        assert model.is_fitted

    def test_should_raise_if_model_not_fitted(self) -> None:
        model = BaselineRegressor()
        predict_data = Table({"feat": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        with pytest.raises(ModelNotFittedError):
            model.predict(predict_data)

    def test_should_raise_if_predict_data_has_differing_features(self) -> None:
        model = BaselineRegressor()
        fit_data = Table({"feat": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        predict_data = Table({"other": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        model = model.fit(fit_data)
        with pytest.raises(FeatureDataMismatchError):
            model.predict(predict_data)

    def test_should_raise_if_predict_data_misses_target_column(self) -> None:
        model = BaselineRegressor()
        fit_data = Table({"feat": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        predict_data = Table({"feat": [0, 1], "other": [0, 1]}).to_tabular_dataset("other")
        model = model.fit(fit_data)
        with pytest.raises(TargetDataMismatchError):
            model.predict(predict_data)

    def test_check_predict_return_type_and_values(self) -> None:
        model = BaselineRegressor()
        data = Table({"feat": [0, 1], "target": [0, 1]}).to_tabular_dataset("target")
        model = model.fit(data)
        result = model.predict(data)
        assert isinstance(result, dict)
        assert result.get("coefficient_of_determination", float("-inf")) >= float("-inf")
        assert result.get("mean_absolute_error", float("inf")) <= float("inf")
        assert result.get("mean_squared_error", float("inf")) <= float("inf")
        assert result.get("median_absolute_deviation", float("inf")) <= float("inf")
