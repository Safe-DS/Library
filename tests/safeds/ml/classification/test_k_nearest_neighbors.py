import pytest

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.exceptions import PredictionError
from safeds.ml.classification import KNearestNeighbors as KNearestNeighborsClassifier
from tests.fixtures import resolve_resource_path


class TestFit:
    def test_k_nearest_neighbors_fit(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors.csv"))
        tagged_table = TaggedTable(table, "T")
        k_nearest_neighbors = KNearestNeighborsClassifier(2)
        k_nearest_neighbors.fit(tagged_table)
        assert True  # This asserts that the fit method succeeds

    def test_k_nearest_neighbors_fit_invalid(self) -> None:
        table = Table.from_csv(
            resolve_resource_path("test_k_nearest_neighbors_invalid.csv")
        )
        tagged_table = TaggedTable(table, "T")
        k_nearest_neighbors = KNearestNeighborsClassifier(2)
        with pytest.raises(LearningError):
            k_nearest_neighbors.fit(tagged_table)


class TestPredict:
    def test_k_nearest_neighbors_predict(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors.csv"))
        tagged_table = TaggedTable(table, "T")
        k_nearest_neighbors = KNearestNeighborsClassifier(2)
        k_nearest_neighbors.fit(tagged_table)
        k_nearest_neighbors.predict(tagged_table.features)
        assert True  # This asserts that the predict method succeeds

    def test_k_nearest_neighbors_predict_not_fitted(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors.csv"))
        tagged_table = TaggedTable(table, "T")
        k_nearest_neighbors = KNearestNeighborsClassifier(2)
        with pytest.raises(PredictionError):
            k_nearest_neighbors.predict(tagged_table.features)

    def test_k_nearest_neighbors_predict_invalid(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_k_nearest_neighbors.csv"))
        invalid_table = Table.from_csv(
            resolve_resource_path("test_k_nearest_neighbors_invalid.csv")
        )
        tagged_table = TaggedTable(table, "T")
        invalid_tagged_table = TaggedTable(invalid_table, "T")
        k_nearest_neighbors = KNearestNeighborsClassifier(2)
        k_nearest_neighbors.fit(tagged_table)
        with pytest.raises(PredictionError):
            k_nearest_neighbors.predict(invalid_tagged_table.features)
