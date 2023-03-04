import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.classification import DecisionTree


def test_decision_tree_fit() -> None:
    table = Table.from_csv("tests/resources/test_decision_tree.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    decision_tree = DecisionTree()
    decision_tree.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_decision_tree_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_decision_tree_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    decision_tree = DecisionTree()
    with pytest.raises(LearningError):
        decision_tree.fit(supervised_dataset)
