import pytest
from safeds.data.tabular import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import PredictionError
from safeds.ml.classification import AdaBoost


def test_ada_boost_predict() -> None:
    table = Table.from_csv("tests/resources/test_ada_boost.csv")
    tagged_table = TaggedTable(table, "T")
    ada_boost = AdaBoost()
    ada_boost.fit(tagged_table)
    ada_boost.predict(tagged_table.feature_vectors)
    assert True  # This asserts that the predict method succeeds


def test_ada_boost_predict_not_fitted() -> None:
    table = Table.from_csv("tests/resources/test_ada_boost.csv")
    tagged_table = TaggedTable(table, "T")
    ada_boost = AdaBoost()
    with pytest.raises(PredictionError):
        ada_boost.predict(tagged_table.feature_vectors)


def test_ada_boost_predict_invalid() -> None:
    table = Table.from_csv("tests/resources/test_ada_boost.csv")
    invalid_table = Table.from_csv("tests/resources/test_ada_boost_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    invalid_tagged_table = TaggedTable(invalid_table, "T")
    ada_boost = AdaBoost()
    ada_boost.fit(tagged_table)
    with pytest.raises(PredictionError):
        ada_boost.predict(invalid_tagged_table.feature_vectors)
