import pandas as pd
from safeds.data.tabular import Column


def test_column_property_all_positive() -> None:
    column = Column(pd.Series([1, 1, 1]), "col1")
    assert column.all(lambda value: value == 1)


def test_column_property_all_negative() -> None:
    column = Column(pd.Series([1, 2, 1]), "col1")
    assert not column.all(lambda value: value == 1)


def test_column_property_any_positive() -> None:
    column = Column(pd.Series([1, 2, 1]), "col1")
    assert column.any(lambda value: value == 1)


def test_column_property_any_negative() -> None:
    column = Column(pd.Series([1, 2, 1]), "col1")
    assert not column.any(lambda value: value == 3)


def test_column_property_none_positive() -> None:
    column = Column(pd.Series([1, 2, 1]), "col1")
    assert column.none(lambda value: value == 3)


def test_column_property_none_negative() -> None:
    column = Column(pd.Series([1, 2, 1]), "col1")
    assert not column.none(lambda value: value == 1)
