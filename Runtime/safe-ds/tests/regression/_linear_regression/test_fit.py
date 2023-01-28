import pytest
from safeds.data import SupervisedDataset, Table
from safeds.exceptions import LearningError
from safeds.regression import LinearRegression


def test_linear_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    linear_regression = LinearRegression()
    linear_regression.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_linear_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    linear_regression = LinearRegression()
    with pytest.raises(LearningError):
        linear_regression.fit(supervised_dataset)
