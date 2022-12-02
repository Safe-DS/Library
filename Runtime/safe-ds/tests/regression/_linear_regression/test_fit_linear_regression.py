import pytest
from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import LearningError
from safe_ds.regression import LinearRegression


def test_linear_regression_fit():
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LinearRegression()
    log_regression.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_linear_regression_fit_invalid():
    table = Table.from_csv("tests/resources/test_linear_regression_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    log_regression = LinearRegression()
    with pytest.raises(LearningError):
        log_regression.fit(supervised_dataset)
