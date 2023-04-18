import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import ColumnLengthMismatchError, NonNumericColumnError


def test_correlation_with() -> None:
    column1 = Column("A", [1, 2, 3, 4])
    column2 = Column("B", [2, 3, 4, 5])
    actual_corr = column1.correlation_with(column2)
    expected_corr = column1._data.corr(column2._data)
    assert actual_corr == expected_corr


def test_correlation_with_raises_if_column_is_not_numeric() -> None:
    column1 = Column("A", [1, 2, 3, 4])
    column2 = Column("B", ["a", "b", "c", "d"])
    with pytest.raises(NonNumericColumnError):
        column1.correlation_with(column2)


def test_correlation_with_raises_if_column_lengths_differ() -> None:
    column1 = Column("A", [1, 2, 3, 4])
    column2 = Column("B", [2])
    with pytest.raises(ColumnLengthMismatchError):
        column1.correlation_with(column2)
