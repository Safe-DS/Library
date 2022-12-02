import pandas as pd
import pytest
from safe_ds.data import Column, ColumnType
from safe_ds.exceptions import IndexOutOfBoundsError


def test_get_value_valid():
    column = Column(
        pd.Series([0, "1"]),
        "testColumn",
        ColumnType.from_numpy_dtype(pd.Series([0, "1"]).dtype),
    )
    assert column.get_value(0) == 0
    assert column.get_value(1) == "1"


def test_get_value_invalid():
    column = Column(
        pd.Series([0, "1"]),
        "testColumn",
        ColumnType.from_numpy_dtype(pd.Series([0, "1"]).dtype),
    )
    with pytest.raises(IndexOutOfBoundsError):
        column.get_value(-1)

    with pytest.raises(IndexOutOfBoundsError):
        column.get_value(2)
