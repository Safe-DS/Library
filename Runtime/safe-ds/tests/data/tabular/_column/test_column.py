import pandas as pd
from safeds.data.tabular import Column


def test_from_columns() -> None:
    column1 = Column(pd.Series([1, 4]), "A")
    column2 = Column(pd.Series([2, 5]), "B")

    assert column1._type == column2._type


def negative_test_from_columns() -> None:
    column1 = Column(pd.Series([1, 4]), "A")
    column2 = Column(pd.Series(["2", "5"]), "B")

    assert column1._type != column2._type
