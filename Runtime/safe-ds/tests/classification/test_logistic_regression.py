import pytest
from safe_ds.classification import LogisticRegression
from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import LearningError, PredictionError


def test_logistic_regression_fit():
    table = Table.from_csv("tests/resources/test_logistic_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_logistic_regression_predict():
    table = Table.from_csv("tests/resources/test_logistic_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(supervised_dataset)
    log_regression.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_logistic_regression_fit_invalid():
    table = Table.from_csv("tests/resources/test_logistic_regression_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    with pytest.raises(LearningError):
        log_regression.fit(supervised_dataset)


def test_logistic_regression_predict_not_fitted():
    table = Table.from_csv("tests/resources/test_logistic_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    with pytest.raises(PredictionError):
        log_regression.predict(supervised_dataset.feature_vectors)


def test_logistic_regression_predict_invalid():
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


def test_logistic_regression_predict_invalid_target_predictions():
    table = Table.from_csv(
        "tests/resources/test_logistic_regression_invalid_target_predictions.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        log_regression.predict(supervised_dataset.feature_vectors)
