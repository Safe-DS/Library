import pytest
from safeds.data.tabular.containers import Table, TaggedTable
from safeds.exceptions import PredictionError
from safeds.ml.regression import ElasticNetRegression
from tests.fixtures import resolve_resource_path


def test_elastic_net_regression_predict() -> None:
    table = Table.from_csv(resolve_resource_path("test_elastic_net_regression.csv"))
    tagged_table = TaggedTable(table, "T")
    en_regression = ElasticNetRegression()
    en_regression.fit(tagged_table)
    en_regression.predict(tagged_table.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_elastic_net_regression_predict_not_fitted() -> None:
    table = Table.from_csv(resolve_resource_path("test_elastic_net_regression.csv"))
    tagged_table = TaggedTable(table, "T")
    en_regression = ElasticNetRegression()
    with pytest.raises(PredictionError):
        en_regression.predict(tagged_table.feature_vectors)


def test_elastic_net_regression_predict_invalid() -> None:
    table = Table.from_csv(resolve_resource_path("test_elastic_net_regression.csv"))
    invalid_table = Table.from_csv(
        resolve_resource_path("test_elastic_net_regression_invalid.csv")
    )
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    en_regression = ElasticNetRegression()
    en_regression.fit(tagged_table)
    with pytest.raises(PredictionError):
        en_regression.predict(invalid_tagged_table.feature_vectors)
