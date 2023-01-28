import pytest
from safeds.data import SupervisedDataset, Table
from safeds.exceptions import LearningError
from safeds.regression import ElasticNetRegression


def test_elastic_net_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_elastic_net_regression.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    en_regression = ElasticNetRegression()
    en_regression.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_elastic_net_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_elastic_net_regression_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    en_regression = ElasticNetRegression()
    with pytest.raises(LearningError):
        en_regression.fit(supervised_dataset)
