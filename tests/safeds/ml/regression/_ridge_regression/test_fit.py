import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.ml.regression import RidgeRegression
from tests.fixtures import resolve_resource_path


def test_ridge_regression_fit() -> None:
    table = Table.from_csv(resolve_resource_path("test_ridge_regression.csv"))
    tagged_table = TaggedTable(table, "T")
    ridge_regression = RidgeRegression()
    ridge_regression.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_ridge_regression_fit_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_ridge_regression_invalid.csv"))
    tagged_table = TaggedTable(table, "T")
    ridge_regression = RidgeRegression()
    with pytest.raises(LearningError):
        ridge_regression.fit(tagged_table)
