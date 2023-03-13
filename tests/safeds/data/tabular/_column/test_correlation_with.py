import pandas as pd
import pytest
from safeds.data.tabular import Column
from safeds.exceptions import ColumnLengthMismatchError, NonNumericColumnError


def test_correlation_with() -> None:
    column1 = Column(pd.Series([1, 2, 3, 4]), "A")
    column2 = Column(pd.Series([2, 3, 4, 5]), "B")
    actual_corr = column1.correlation_with(column2)
    expected_corr = column1._data.corr(column2._data)
    assert actual_corr == expected_corr


def test_correlation_with_NonNumericColumnError() -> None:
    column1 = Column(pd.Series([1, 2, 3, 4]), "A")
    column2 = Column(pd.Series(["a", "b", "c", "d"]), "B")
    with pytest.raises(NonNumericColumnError):
        column1.correlation_with(column2)


def test_correlation_with_ColumnsLengthMismachtError() -> None:
    column1 = Column(pd.Series([1, 2, 3, 4]), "A")
    column2 = Column(pd.Series([2]), "B")
    with pytest.raises(ColumnLengthMismatchError):
        column1.correlation_with(column2)
