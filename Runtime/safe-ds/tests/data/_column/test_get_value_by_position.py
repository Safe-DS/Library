import pandas as pd
import pytest
from safe_ds.data import Column
from safe_ds.exceptions import IndexOutOfBoundsError


def test_value_by_position_valid():
    column = Column(pd.Series([0, "1"]), "testColumn", pd.Series([0, "1"]).dtype)
    assert column.get_value_by_position(0) == 0
    assert column.get_value_by_position(1) == "1"


def test_value_by_position_invalid():
    column = Column(pd.Series([0, "1"]), "testColumn", pd.Series([0, "1"]).dtype)
    with pytest.raises(IndexOutOfBoundsError):
        column.get_value_by_position(-1)

    with pytest.raises(IndexOutOfBoundsError):
        column.get_value_by_position(2)
