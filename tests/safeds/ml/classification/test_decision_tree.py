import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError, PredictionError
from safeds.ml.classification import DecisionTree
from tests.fixtures import resolve_resource_path


class TestFit:
    def test_decision_tree_fit(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_decision_tree.csv"))
        tagged_table = TaggedTable(table, "T")
        decision_tree = DecisionTree()
        decision_tree.fit(tagged_table)
        assert True  # This asserts that the fit method succeeds

    def test_decision_tree_fit_invalid(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_decision_tree_invalid.csv"))
        tagged_table = TaggedTable(table, "T")
        decision_tree = DecisionTree()
        with pytest.raises(LearningError):
            decision_tree.fit(tagged_table)


class TestPredict:
    def test_decision_tree_predict(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_decision_tree.csv"))
        tagged_table = TaggedTable(table, "T")
        decision_tree = DecisionTree()
        decision_tree.fit(tagged_table)
        decision_tree.predict(tagged_table.features)
        assert True  # This asserts that the predict method succeeds

    def test_decision_tree_predict_not_fitted(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_decision_tree.csv"))
        tagged_table = TaggedTable(table, "T")
        decision_tree = DecisionTree()
        with pytest.raises(PredictionError):
            decision_tree.predict(tagged_table.features)

    def test_decision_tree_predict_invalid(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_decision_tree.csv"))
        invalid_table = Table.from_csv(
            resolve_resource_path("test_decision_tree_invalid.csv")
        )
        tagged_table = TaggedTable(table, "T")
        invalid_tagged_table = TaggedTable(invalid_table, "T")
        decision_tree = DecisionTree()
        decision_tree.fit(tagged_table)
        with pytest.raises(PredictionError):
            decision_tree.predict(invalid_tagged_table.features)
