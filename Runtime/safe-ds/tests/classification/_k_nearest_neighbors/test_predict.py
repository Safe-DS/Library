import pytest
from safe_ds.classification import KNearestNeighbors as KNearestNeighborsClassifier
from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import PredictionError


def test_k_nearest_neighbors_predict() -> None:
    table = Table.from_csv("tests/resources/test_k_nearest_neighbors.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    k_nearest_neighbors = KNearestNeighborsClassifier(2)
    k_nearest_neighbors.fit(supervised_dataset)
    k_nearest_neighbors.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_k_nearest_neighbors_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_k_nearest_neighbors.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    k_nearest_neighbors = KNearestNeighborsClassifier(2)
    with pytest.raises(PredictionError):
        k_nearest_neighbors.predict(supervised_dataset.feature_vectors)


def test_k_nearest_neighbors_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_k_nearest_neighbors.csv")
    invalid_table = Table.from_csv(
        "tests/resources/test_k_nearest_neighbors_invalid.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    invalid_supervised_dataset = SupervisedDataset(invalid_table, "T")
    k_nearest_neighbors = KNearestNeighborsClassifier(2)
    k_nearest_neighbors.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        k_nearest_neighbors.predict(invalid_supervised_dataset.feature_vectors)
