import pytest
from safeds.data.tabular import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.regression import ElasticNetRegression


def test_elastic_net_regression_fit() -> None:
    table = Table.from_csv("tests/resources/test_elastic_net_regression.csv")
    tagged_table = TaggedTable(table, "T")
    en_regression = ElasticNetRegression()
    en_regression.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_elastic_net_regression_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_elastic_net_regression_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    en_regression = ElasticNetRegression()
    with pytest.raises(LearningError):
        en_regression.fit(tagged_table)
