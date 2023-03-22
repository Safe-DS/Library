import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import PredictionError
from safeds.ml.classification import RandomForest as RandomForestClassifier


def test_random_forest_predict() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    tagged_table = TaggedTable(table, "T")
    random_forest = RandomForestClassifier()
    random_forest.fit(tagged_table)
    random_forest.predict(tagged_table.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_random_forest_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    tagged_table = TaggedTable(table, "T")
    random_forest = RandomForestClassifier()
    with pytest.raises(PredictionError):
        random_forest.predict(tagged_table.feature_vectors)


def test_random_forest_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    invalid_table = Table.from_csv("tests/resources/test_random_forest_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    random_forest = RandomForestClassifier()
    random_forest.fit(tagged_table)
    with pytest.raises(PredictionError):
        random_forest.predict(invalid_tagged_table.feature_vectors)
