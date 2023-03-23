import pytest

from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.ml.regression import GradientBoosting


def test_gradient_boosting_regression_fit() -> None:
    table = Table.from_csv(resolve_resource_path("test_gradient_boosting_regression.csv"))
    tagged_table = TaggedTable(table, "T")
    gradient_boosting = GradientBoosting()
    gradient_boosting.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_gradient_boosting_regression_fit_invalid() -> None:
    table = Table.from_csv(
        resolve_resource_path("test_gradient_boosting_regression_invalid.csv")
    )
    tagged_table = TaggedTable(table, "T")
    gradient_boosting_regression = GradientBoosting()
    with pytest.raises(LearningError):
        gradient_boosting_regression.fit(tagged_table)
