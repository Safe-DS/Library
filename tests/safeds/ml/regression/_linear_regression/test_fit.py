import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.ml.regression import LinearRegression
from tests.fixtures import resolve_resource_path


def test_linear_regression_fit() -> None:
    table = Table.from_csv(resolve_resource_path("test_linear_regression.csv"))
    tagged_table = TaggedTable(table, "T")
    linear_regression = LinearRegression()
    linear_regression.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_linear_regression_fit_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_linear_regression_invalid.csv"))
    tagged_table = TaggedTable(table, "T")
    linear_regression = LinearRegression()
    with pytest.raises(LearningError):
        linear_regression.fit(tagged_table)
