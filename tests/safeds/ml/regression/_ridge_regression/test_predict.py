import pytest
from safeds.data import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import PredictionError
from safeds.ml.regression import RidgeRegression


def test_ridge_regression_predict() -> None:
    table = Table.from_csv("tests/resources/test_ridge_regression.csv")
    tagged_table = TaggedTable(table, "T")
    ridge_regression = RidgeRegression()
    ridge_regression.fit(tagged_table)
    ridge_regression.predict(tagged_table.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_ridge_regression_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_ridge_regression.csv")
    tagged_table = TaggedTable(table, "T")
    ridge_regression = RidgeRegression()
    with pytest.raises(PredictionError):
        ridge_regression.predict(tagged_table.feature_vectors)


def test_ridge_regression_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_ridge_regression.csv")
    invalid_table = Table.from_csv("tests/resources/test_ridge_regression_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    ridge_regression = RidgeRegression()
    ridge_regression.fit(tagged_table)
    with pytest.raises(PredictionError):
        ridge_regression.predict(invalid_tagged_table.feature_vectors)
