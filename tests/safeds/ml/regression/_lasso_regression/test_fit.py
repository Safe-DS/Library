import pytest
from safeds.data.tabular import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.regression import LassoRegression


def test_lasso_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_lasso_regression.csv")
    tagged_table = TaggedTable(table, "T")
    lasso_regression = LassoRegression()
    lasso_regression.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_lasso_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_lasso_regression_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    lasso_regression = LassoRegression()
    with pytest.raises(LearningError):
        lasso_regression.fit(tagged_table)
