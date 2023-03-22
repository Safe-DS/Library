import pytest
from safeds.data import TaggedTable
from safeds.data.tabular import Table
from safeds.exceptions import LearningError
from safeds.ml.classification import AdaBoost


def test_ada_boost_fit() -> None:
    table = Table.from_csv("tests/resources/test_ada_boost.csv")
    tagged_table = TaggedTable(table, "T")
    ada_boost = AdaBoost()
    ada_boost.fit(tagged_table)
    assert True  # This asserts that the fit method succeeds


def test_ada_boost_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_ada_boost_invalid.csv")
    tagged_table = TaggedTable(table, "T")
    ada_boost = AdaBoost()
    with pytest.raises(LearningError):
        ada_boost.fit(tagged_table)
