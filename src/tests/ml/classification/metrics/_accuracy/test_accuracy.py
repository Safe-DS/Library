import pandas as pd
from safeds.data.tabular import Column
from safeds.ml.classification.metrics import accuracy


def test_accuracy() -> None:
    c1 = Column(pd.Series(data=[1, 2, 3, 4]), "TestColumn1")
    c2 = Column(pd.Series(data=[1, 2, 3, 3]), "TestColumn2")
    assert accuracy(c1, c2) == 0.75


def test_accuracy_different_types() -> None:
    c1 = Column(pd.Series(data=["1", "2", "3", "4"]), "TestColumn1")
    c2 = Column(pd.Series(data=[1, 2, 3, 3]), "TestColumn2")
    assert accuracy(c1, c2) == 0.0
