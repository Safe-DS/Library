import pytest

from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import LearningError
from safeds.exceptions import PredictionError
from safeds.ml.classification import RandomForest as RandomForestClassifier
from tests.fixtures import resolve_resource_path


class TestFit:
    def test_logistic_regression_fit(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_random_forest.csv"))
        tagged_table = TaggedTable(table, "T")
        random_forest = RandomForestClassifier()
        random_forest.fit(tagged_table)
        assert True  # This asserts that the fit method succeeds

    def test_logistic_regression_fit_invalid(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_random_forest_invalid.csv"))
        tagged_table = TaggedTable(table, "T")
        random_forest = RandomForestClassifier()
        with pytest.raises(LearningError):
            random_forest.fit(tagged_table)


class TestPredict:
    def test_random_forest_predict(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_random_forest.csv"))
        tagged_table = TaggedTable(table, "T")
        random_forest = RandomForestClassifier()
        random_forest.fit(tagged_table)
        random_forest.predict(tagged_table.features)
        assert True  # This asserts that the predict method succeeds

    def test_random_forest_predict_not_fitted(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_random_forest.csv"))
        tagged_table = TaggedTable(table, "T")
        random_forest = RandomForestClassifier()
        with pytest.raises(PredictionError):
            random_forest.predict(tagged_table.features)

    def test_random_forest_predict_invalid(self) -> None:
        table = Table.from_csv(resolve_resource_path("test_random_forest.csv"))
        invalid_table = Table.from_csv(
            resolve_resource_path("test_random_forest_invalid.csv")
        )
        tagged_table = TaggedTable(table, "T")
        invalid_tagged_table = TaggedTable(invalid_table, "T")
        random_forest = RandomForestClassifier()
        random_forest.fit(tagged_table)
        with pytest.raises(PredictionError):
            random_forest.predict(invalid_tagged_table.features)
