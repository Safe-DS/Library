import pytest
from safeds.data.tabular import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.regression import LinearRegression


def test_linear_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    tagged_table = TaggedTable(table, "T")
    linear_regression = LinearRegression()
    linear_regression.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_linear_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    linear_regression = LinearRegression()
    with pytest.raises(LearningError):
        linear_regression.fit(tagged_table)
