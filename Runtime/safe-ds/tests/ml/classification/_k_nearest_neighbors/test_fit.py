import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.classification import KNearestNeighbors as KNearestNeighborsClassifier


def test_k_nearest_neighbors_fit() -> None:
    table = Table.from_csv("tests/resources/test_k_nearest_neighbors.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    k_nearest_neighbors = KNearestNeighborsClassifier(2)
    k_nearest_neighbors.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_k_nearest_neighbors_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_k_nearest_neighbors_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    k_nearest_neighbors = KNearestNeighborsClassifier(2)
    with pytest.raises(LearningError):
        k_nearest_neighbors.fit(supervised_dataset)
