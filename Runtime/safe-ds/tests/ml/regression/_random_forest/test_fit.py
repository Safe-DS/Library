import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.regression import RandomForest as RandomForestRegressor


def test_random_forest_fit() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    random_forest = RandomForestRegressor()
    random_forest.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_random_forest_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_random_forest_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    random_forest = RandomForestRegressor()
    with pytest.raises(LearningError):
        random_forest.fit(supervised_dataset)
