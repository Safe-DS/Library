import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import PredictionError
from safeds.ml.regression import RandomForest as RandomForestRegressor
from tests.fixtures import resolve_resource_path


def test_random_forest_predict() -> None:
    table = Table.from_csv(resolve_resource_path("test_random_forest.csv"))
    tagged_table = TaggedTable(table, "T")
    random_forest = RandomForestRegressor()
    random_forest.fit(tagged_table)
    random_forest.predict(tagged_table.features)
    assert True  # This asserts that the predict method succeeds


def test_random_forest_predict_not_fitted() -> None:
    table = Table.from_csv(resolve_resource_path("test_random_forest.csv"))
    tagged_table = TaggedTable(table, "T")
    random_forest = RandomForestRegressor()
    with pytest.raises(PredictionError):
        random_forest.predict(tagged_table.features)


def test_random_forest_predict_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_random_forest.csv"))
    invalid_table = Table.from_csv(
        resolve_resource_path("test_random_forest_invalid.csv")
    )
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    random_forest = RandomForestRegressor()
    random_forest.fit(tagged_table)
    with pytest.raises(PredictionError):
        random_forest.predict(invalid_tagged_table.features)
