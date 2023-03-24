import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import PredictionError
from safeds.ml.regression import KNearestNeighbors as KNearestNeighborsRegressor
from tests.fixtures import resolve_resource_path


def test_k_nearest_neighbors_predict() -> None:
    table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors.csv"))
    tagged_table = TaggedTable(table, "T")
    k_nearest_neighbors = KNearestNeighborsRegressor(2)
    k_nearest_neighbors.fit(tagged_table)
    k_nearest_neighbors.predict(tagged_table.features)
    assert True  # This asserts that the predict method succeeds


def test_k_nearest_neighbors_predict_not_fitted() -> None:
    table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors.csv"))
    tagged_table = TaggedTable(table, "T")
    k_nearest_neighbors = KNearestNeighborsRegressor(2)
    with pytest.raises(PredictionError):
        k_nearest_neighbors.predict(tagged_table.features)


def test_k_nearest_neighbors_predict_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors.csv"))
    invalid_table = Table.from_csv(
        resolve_resource_path("test_k_nearest_neighbors_invalid.csv")
    )
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    k_nearest_neighbors = KNearestNeighborsRegressor(2)
    k_nearest_neighbors.fit(tagged_table)
    with pytest.raises(PredictionError):
        k_nearest_neighbors.predict(invalid_tagged_table.features)
