import pytest
from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import PredictionError
from safe_ds.regression import DecisionTree


def test_decision_tree_predict() -> None:
    table = Table.from_csv("tests/resources/test_decision_tree.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    decision_tree = DecisionTree()
    decision_tree.fit(supervised_dataset)
    decision_tree.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_decision_tree_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_decision_tree.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    decision_tree = DecisionTree()
    with pytest.raises(PredictionError):
        decision_tree.predict(supervised_dataset.feature_vectors)


def test_decision_tree_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_decision_tree.csv")
    invalid_table = Table.from_csv("tests/resources/test_decision_tree_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    invalid_supervised_dataset = SupervisedDataset(invalid_table, "T")
    decision_tree = DecisionTree()
    decision_tree.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        decision_tree.predict(invalid_supervised_dataset.feature_vectors)
