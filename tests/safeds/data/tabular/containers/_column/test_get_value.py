import pandas as pd
import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import IndexOutOfBoundsError


def test_get_value_valid() -> None:
    column = Column("testColumn", pd.Series([0, "1"]))
    assert column.get_value(0) == 0
    assert column.get_value(1) == "1"


def test_get_value_invalid() -> None:
    column = Column("testColumn", pd.Series([0, "1"]))
    with pytest.raises(IndexOutOfBoundsError):
        column.get_value(-1)

    with pytest.raises(IndexOutOfBoundsError):
        column.get_value(2)
