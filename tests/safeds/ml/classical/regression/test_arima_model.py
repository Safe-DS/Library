from typing import Any

import pytest
from safeds.data.labeled.containers import TimeSeriesDataset
from safeds.data.tabular.containers import Table
from safeds.exceptions import (
    DatasetMissesDataError,
    MissingValuesColumnError,
    ModelNotFittedError,
    NonNumericColumnError,
    NonTimeSeriesDatasetError,
)
from safeds.ml.classical.regression import ArimaModelRegressor, LassoRegressor

from tests.helpers import resolve_resource_path


def test_arima_model() -> None:
    # Create a DataFrame
    _inflation_path = "_datas/US_Inflation_rates.csv"
    time_series = Table.from_csv_file(
        path=resolve_resource_path(_inflation_path),
    )
    train_ts, test_ts = time_series.split_rows(0.8)
    model = ArimaModelRegressor()
    trained_model = model.fit(train_ts.to_time_series_dataset("value", "date"))
    trained_model.predict(test_ts.to_time_series_dataset("value", "date"))
    # suggest it ran through
    assert True


def create_test_data() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        {"time": [1, 2, 3, 4, 5, 6, 7, 8, 9], "value": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
        time_name="time",
        target_name="value",
    )


def create_test_data_with_feature() -> TimeSeriesDataset:
    return TimeSeriesDataset(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        time_name="time",
        target_name="value",
    )


def test_should_succeed_on_valid_data() -> None:
    valid_data = create_test_data()
    model = ArimaModelRegressor()
    model.fit(valid_data)
    assert True


def test_should_not_change_input_regressor() -> None:
    valid_data = create_test_data()
    model = ArimaModelRegressor()
    model.fit(valid_data)
    assert not model.is_fitted


def test_should_not_change_input_table() -> None:
    valid_data = create_test_data()
    valid_data_copy = create_test_data()
    model = ArimaModelRegressor()
    model.fit(valid_data)
    assert valid_data_copy == valid_data


def test_should_succeed_on_valid_data_plot() -> None:
    valid_data = create_test_data()
    model = ArimaModelRegressor()
    fitted_model = model.fit(valid_data)
    fitted_model.plot_predictions(valid_data)
    assert True


@pytest.mark.parametrize(
    ("invalid_data", "expected_error", "expected_error_msg"),
    [
        (
            Table(
                {
                    "id": [1, 4],
                    "feat1": [1, 5],
                    "feat2": [3, 6],
                    "target": ["0", 1],
                },
            ).to_time_series_dataset(target_name="target", time_name="id"),
            NonNumericColumnError,
            r"Tried to do a numerical operation on one or multiple non-numerical columns: \ntarget",
        ),
        (
            Table(
                {
                    "id": [1, 4],
                    "feat1": [1, 5],
                    "feat2": [3, 6],
                    "target": [None, 1],
                },
            ).to_time_series_dataset(target_name="target", time_name="id"),
            MissingValuesColumnError,
            r"Tried to do an operation on one or multiple columns containing missing values: \ntarget\nYou can use the Imputer to replace the missing values based on different strategies.\nIf you want toremove the missing values entirely you can use the method `TimeSeries.remove_rows_with_missing_values`.",
        ),
        (
            Table(
                {
                    "id": [],
                    "feat1": [],
                    "feat2": [],
                    "target": [],
                },
            ).to_time_series_dataset(target_name="target", time_name="id"),
            DatasetMissesDataError,
            r"Dataset contains no rows",
        ),
    ],
    ids=["non-numerical data", "missing values in data", "no rows in data"],
)
def test_should_raise_on_invalid_data(
    invalid_data: TimeSeriesDataset,
    expected_error: Any,
    expected_error_msg: str,
) -> None:
    model = ArimaModelRegressor()
    with pytest.raises(expected_error, match=expected_error_msg):
        model.fit(invalid_data)


@pytest.mark.parametrize(
    "table",
    [
        Table(
            {
                "a": [1.0, 0.0, 0.0, 0.0],
                "b": [0.0, 1.0, 1.0, 0.0],
                "c": [0.0, 0.0, 0.0, 1.0],
            },
        ),
    ],
    ids=["table"],
)
def test_should_raise_if_given_normal_table(table: Table) -> None:
    model = ArimaModelRegressor()
    with pytest.raises(NonTimeSeriesDatasetError):
        model.fit(table)  # type: ignore[arg-type]


def test_correct_structure_of_time_series_with_features() -> None:
    data = create_test_data_with_feature()
    model = ArimaModelRegressor()
    model = model.fit(data)
    predics_ts = model.predict(data)
    assert len(predics_ts.time) == len(data.time)
    assert len(predics_ts.target) == len(data.target)
    assert predics_ts.time.name == data.time.name
    assert predics_ts.target.name == data.target.name + " " + "forecasted"
    assert predics_ts.features.column_names == data.features.column_names


def test_correct_structure_of_time_series() -> None:
    data = create_test_data()
    model = ArimaModelRegressor()
    model = model.fit(data)
    predics_ts = model.predict(data)
    assert len(predics_ts.time) == len(data.time)
    assert len(predics_ts.target) == len(data.target)
    assert predics_ts.time.name == data.time.name
    assert predics_ts.target.name == data.target.name + " " + "forecasted"
    assert predics_ts.features.column_names == data.features.column_names


def test_should_raise_if_not_fitted() -> None:
    model = ArimaModelRegressor()
    with pytest.raises(ModelNotFittedError):
        model.predict(create_test_data())


def test_if_fitted_not_fitted() -> None:
    model = ArimaModelRegressor()
    assert not model.is_fitted


def test_if_fitted_fitted() -> None:
    model = ArimaModelRegressor()
    model = model.fit(create_test_data())
    assert model.is_fitted


def test_should_raise_if_horizon_too_small_plot() -> None:
    model = ArimaModelRegressor()
    with pytest.raises(ModelNotFittedError):
        model.plot_predictions(create_test_data())


def test_should_return_same_hash_for_equal_regressor() -> None:
    regressor1 = ArimaModelRegressor()
    regressor2 = ArimaModelRegressor()
    assert hash(regressor1) == hash(regressor2)


def test_should_return_different_hash_for_unequal_regressor() -> None:
    regressor1 = ArimaModelRegressor()
    regressor2 = LassoRegressor()
    assert hash(regressor1) != hash(regressor2)


def test_should_return_different_hash_for_same_regressor_fit() -> None:
    regressor1 = ArimaModelRegressor()
    regressor1_fit = regressor1.fit(create_test_data())
    assert hash(regressor1) != hash(regressor1_fit)


def test_should_return_different_hash_for_regressor_fit() -> None:
    regressor1 = ArimaModelRegressor()
    regressor2 = ArimaModelRegressor()
    regressor1_fit = regressor1.fit(create_test_data())
    assert hash(regressor1_fit) != hash(regressor2)
