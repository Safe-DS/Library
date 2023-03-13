import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.classification import LogisticRegression


def test_logistic_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_logistic_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    log_regression.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_logistic_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_logistic_regression_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LogisticRegression()
    with pytest.raises(LearningError):
        log_regression.fit(supervised_dataset)
