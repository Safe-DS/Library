import pytest
from safeds.data.tabular import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import PredictionError
from safeds.ml.regression import LinearRegression


def test_linear_regression_predict() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    tagged_table = TaggedTable(table, "T")
    linear_regression = LinearRegression()
    linear_regression.fit(tagged_table)
    linear_regression.predict(tagged_table.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_linear_regression_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    tagged_table = TaggedTable(table, "T")
    linear_regression = LinearRegression()
    with pytest.raises(PredictionError):
        linear_regression.predict(tagged_table.feature_vectors)


def test_linear_regression_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_linear_regression.csv")
    invalid_table = Table.from_csv("tests/resources/test_linear_regression_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    linear_regression = LinearRegression()
    linear_regression.fit(tagged_table)
    with pytest.raises(PredictionError):
        linear_regression.predict(invalid_tagged_table.feature_vectors)
