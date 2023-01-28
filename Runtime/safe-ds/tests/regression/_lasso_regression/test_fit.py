import pytest
from safeds.data import SupervisedDataset, Table
from safeds.exceptions import LearningError
from safeds.regression import LassoRegression


def test_lasso_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_lasso_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    lasso_regression = LassoRegression()
    lasso_regression.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_lasso_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_lasso_regression_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    lasso_regression = LassoRegression()
    with pytest.raises(LearningError):
        lasso_regression.fit(supervised_dataset)
