import pytest
from safeds.data import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.classification import RandomForest as RandomForestClassifier


def test_logistic_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    tagged_table = TaggedTable(table, "T")
    random_forest = RandomForestClassifier()
    random_forest.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_logistic_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_random_forest_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    random_forest = RandomForestClassifier()
    with pytest.raises(LearningError):
        random_forest.fit(tagged_table)
