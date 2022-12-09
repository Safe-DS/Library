import pytest
from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import PredictionError
from safe_ds.regression import LinearRegression


def test_linear_regression_predict() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LinearRegression()
    log_regression.fit(supervised_dataset)
    log_regression.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_linear_regression_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LinearRegression()
    with pytest.raises(PredictionError):
        log_regression.predict(supervised_dataset.feature_vectors)


def test_linear_regression_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    invalid_table = Table.from_csv("tests/resources/test_linear_regression_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    invalid_supervised_dataset = SupervisedDataset(invalid_table, "T")
    log_regression = LinearRegression()
    log_regression.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        log_regression.predict(invalid_supervised_dataset.feature_vectors)


def test_linear_regression_predict_invalid_target_predictions() -> None:
    table = Table.from_csv(
        "tests/resources/test_linear_regression_invalid_target_predictions.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LinearRegression()
    log_regression.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        log_regression.predict(supervised_dataset.feature_vectors)
