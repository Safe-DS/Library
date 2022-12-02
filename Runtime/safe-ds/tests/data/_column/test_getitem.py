# pylint: disable=pointless-statement

import pandas as pd
import pytest
from safe_ds.data import Column, ColumnType
from safe_ds.exceptions import IndexOutOfBoundsError


def test_getitem_valid():
    column = Column(
        pd.Series([0, "1"]),
        "testColumn",
        ColumnType.from_numpy_dtype(pd.Series([0, "1"]).dtype),
    )
    assert column[0] == 0
    assert column[1] == "1"


# noinspection PyStatementEffect
def test_getitem_invalid():
    column = Column(
        pd.Series([0, "1"]),
        "testColumn",
        ColumnType.from_numpy_dtype(pd.Series([0, "1"]).dtype),
    )
    with pytest.raises(IndexOutOfBoundsError):
        column[-1]

    with pytest.raises(IndexOutOfBoundsError):
        column[2]
