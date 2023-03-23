import pytest

from tests.fixtures import resolve_resource_path
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.ml.regression import KNearestNeighbors as KNearestNeighborsRegressor


def test_k_nearest_neighbors_fit() -> None:
    table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors.csv"))
    tagged_table = TaggedTable(table, "T")
    k_nearest_neighbors = KNearestNeighborsRegressor(2)
    k_nearest_neighbors.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_k_nearest_neighbors_fit_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors_invalid.csv"))
    tagged_table = TaggedTable(table, "T")
    k_nearest_neighbors = KNearestNeighborsRegressor(2)
    with pytest.raises(LearningError):
        k_nearest_neighbors.fit(tagged_table)
