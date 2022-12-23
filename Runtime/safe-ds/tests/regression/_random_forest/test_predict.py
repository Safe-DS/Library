import pytest
from safe_ds.data import SupervisedDataset, Table
from safe_ds.exceptions import PredictionError
from safe_ds.regression import RandomForest as RandomForestRegressor


def test_random_forest_predict() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    random_forest = RandomForestRegressor()
    random_forest.fit(supervised_dataset)
    random_forest.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_random_forest_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    random_forest = RandomForestRegressor()
    with pytest.raises(PredictionError):
        random_forest.predict(supervised_dataset.feature_vectors)


def test_random_forest_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_random_forest.csv")
    invalid_table = Table.from_csv("tests/resources/test_random_forest_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    invalid_supervised_dataset = SupervisedDataset(invalid_table, "T")
    random_forest = RandomForestRegressor()
    random_forest.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        random_forest.predict(invalid_supervised_dataset.feature_vectors)


def test_random_forest_predict_invalid_target_predictions() -> None:
    table = Table.from_csv(
        "tests/resources/test_random_forest_invalid_target_predictions.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    random_forest = RandomForestRegressor()
    random_forest.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        random_forest.predict(supervised_dataset.feature_vectors)
