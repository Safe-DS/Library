import pytest
from safe_ds.classification import LogisticRegression
from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import LearningError


def test_logistic_regression_fit():
    table = Table.from_csv("tests/resources/test_logistic_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_logistic_regression_fit_invalid():
    table = Table.from_csv("tests/resources/test_logistic_regression_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    with pytest.raises(LearningError):
        log_regression.fit(supervised_dataset)
