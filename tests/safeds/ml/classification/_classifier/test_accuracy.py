import pandas as pd
from safeds.data.tabular.containers import Column, Table, TaggedTable
from ._dummy_classifier import DummyClassifier


def test_accuracy() -> None:
    c1 = Column(pd.Series(data=[1, 2, 3, 4]), "predicted")
    c2 = Column(pd.Series(data=[1, 2, 3, 3]), "expected")
    table = TaggedTable(Table.from_columns([c1, c2]), target_name="expected")

    assert DummyClassifier().accuracy(table) == 0.75


def test_accuracy_different_types() -> None:
    c1 = Column(pd.Series(data=["1", "2", "3", "4"]), "predicted")
    c2 = Column(pd.Series(data=[1, 2, 3, 3]), "expected")
    table = TaggedTable(Table.from_columns([c1, c2]), target_name="expected")

    assert DummyClassifier().accuracy(table) == 0.0
