import pytest
from safeds.classification import LogisticRegression
from safeds.data import SupervisedDataset, Table
from safeds.exceptions import PredictionError


def test_logistic_regression_predict() -> None:
    table = Table.from_csv("tests/resources/test_logistic_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(supervised_dataset)
    log_regression.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_logistic_regression_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_logistic_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    with pytest.raises(PredictionError):
        log_regression.predict(supervised_dataset.feature_vectors)


def test_logistic_regression_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_logistic_regression.csv")
    invalid_table = Table.from_csv(
        "tests/resources/test_logistic_regression_invalid.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    invalid_supervised_dataset = SupervisedDataset(invalid_table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        log_regression.predict(invalid_supervised_dataset.feature_vectors)
