import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.classification import RandomForest as RandomForestClassifier


def test_logistic_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    random_forest = RandomForestClassifier()
    random_forest.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_logistic_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_random_forest_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    random_forest = RandomForestClassifier()
    with pytest.raises(LearningError):
        random_forest.fit(supervised_dataset)
