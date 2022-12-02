import pandas as pd
from safe_ds.classification.metrics import accuracy
from safe_ds.data import Column
from safe_ds.data._column_type import IntColumnType, StringColumnType


def test_accuracy():
    c1 = Column(
        pd.Series(data=[1, 2, 3, 4]),
        "TestColumn1",
        IntColumnType,
    )
    c2 = Column(
        pd.Series(data=[1, 2, 3, 3]),
        "TestColumn2",
        IntColumnType,
    )
    assert accuracy(c1, c2) == 0.75


def test_accuracy_different_types():
    c1 = Column(
        pd.Series(data=["1", "2", "3", "4"]),
        "TestColumn1",
        StringColumnType,
    )
    c2 = Column(
        pd.Series(data=[1, 2, 3, 3]),
        "TestColumn2",
        IntColumnType,
    )
    assert accuracy(c1, c2) == 0.0
