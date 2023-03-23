import pytest

from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.ml.regression import RandomForest as RandomForestRegressor


def test_random_forest_fit() -> None:
    table = Table.from_csv(resolve_resource_path("test_random_forest.csv"))
    tagged_table = TaggedTable(table, "T")
    random_forest = RandomForestRegressor()
    random_forest.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_random_forest_fit_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_random_forest_invalid.csv"))
    tagged_table = TaggedTable(table, "T")
    random_forest = RandomForestRegressor()
    with pytest.raises(LearningError):
        random_forest.fit(tagged_table)
