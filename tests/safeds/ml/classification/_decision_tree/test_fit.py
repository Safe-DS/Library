import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.ml.classification import DecisionTree


def test_decision_tree_fit() -> None:
    table = Table.from_csv("tests/resources/test_decision_tree.csv")
    tagged_table = TaggedTable(table, "T")
    decision_tree = DecisionTree()
    decision_tree.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_decision_tree_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_decision_tree_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    decision_tree = DecisionTree()
    with pytest.raises(LearningError):
        decision_tree.fit(tagged_table)
