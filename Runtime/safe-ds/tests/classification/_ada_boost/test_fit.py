import pytest
from safeds.classification import AdaBoost
from safeds.data import SupervisedDataset, Table
from safeds.exceptions import LearningError


def test_ada_boost_fit() -> None:
    table = Table.from_csv("tests/resources/test_ada_boost.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    ada_boost = AdaBoost()
    ada_boost.fit(supervised_dataset)
    assert True  # This asserts that the fit method succeeds


def test_ada_boost_fit_invalid() -> None:
    table = Table.from_csv("tests/resources/test_ada_boost_invalid.csv")
    supervised_dataset = SupervisedDataset(table, "T")
    ada_boost = AdaBoost()
    with pytest.raises(LearningError):
        ada_boost.fit(supervised_dataset)
