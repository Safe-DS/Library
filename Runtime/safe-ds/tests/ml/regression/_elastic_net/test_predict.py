import pytest
from safeds.data import SupervisedDataset
from safeds.data.tabular import Table
from safeds.exceptions import PredictionError
from safeds.ml.regression import ElasticNetRegression


def test_elastic_net_regression_predict() -> None:
    table = Table.from_csv("tests/resources/test_elastic_net_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    en_regression = ElasticNetRegression()
    en_regression.fit(supervised_dataset)
    en_regression.predict(supervised_dataset.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_elastic_net_regression_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_elastic_net_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    en_regression = ElasticNetRegression()
    with pytest.raises(PredictionError):
        en_regression.predict(supervised_dataset.feature_vectors)


def test_elastic_net_regression_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_elastic_net_regression.csv")
    invalid_table = Table.from_csv(
        "tests/resources/test_elastic_net_regression_invalid.csv"
    )
    supervised_dataset = SupervisedDataset(table, "T")
    invalid_supervised_dataset = SupervisedDataset(invalid_table, "T")
    en_regression = ElasticNetRegression()
    en_regression.fit(supervised_dataset)
    with pytest.raises(PredictionError):
        en_regression.predict(invalid_supervised_dataset.feature_vectors)
