import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import PredictionError
from safeds.ml.classification import DecisionTree
from tests.fixtures import resolve_resource_path


def test_decision_tree_predict() -> None:
    table = Table.from_csv(resolve_resource_path("test_decision_tree.csv"))
    tagged_table = TaggedTable(table, "T")
    decision_tree = DecisionTree()
    decision_tree.fit(tagged_table)
    decision_tree.predict(tagged_table.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_decision_tree_predict_not_fitted() -> None:
    table = Table.from_csv(resolve_resource_path("test_decision_tree.csv"))
    tagged_table = TaggedTable(table, "T")
    decision_tree = DecisionTree()
    with pytest.raises(PredictionError):
        decision_tree.predict(tagged_table.feature_vectors)


def test_decision_tree_predict_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_decision_tree.csv"))
    invalid_table = Table.from_csv(
        resolve_resource_path("test_decision_tree_invalid.csv")
    )
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    decision_tree = DecisionTree()
    decision_tree.fit(tagged_table)
    with pytest.raises(PredictionError):
        decision_tree.predict(invalid_tagged_table.feature_vectors)
