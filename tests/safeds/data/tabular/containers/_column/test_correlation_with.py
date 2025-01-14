import pytest

from safeds.data.tabular.containers import Column
from safeds.exceptions import (
    ColumnTypeError,
    LengthMismatchError,
    MissingValuesError,
)


@pytest.mark.parametrize(
    ("values_1", "values_2", "expected"),
    [
        ([0, 1, 2], [0, 1, 2], 1.0),
        ([0, 1, 2], [0, -1, -2], -1.0),
    ],
    ids=[
        "positive correlation",
        "negative correlation",
    ],
)
def test_should_return_correlation_between_two_columns(values_1: list, values_2: list, expected: float) -> None:
    column1 = Column("col1", values_1)
    column2 = Column("col2", values_2)
    assert column1.correlation_with(column2) == expected


@pytest.mark.parametrize(
    ("values_1", "values_2"),
    [
        (["a"], [1]),
        ([None], [1]),
        ([1], ["a"]),
        ([1], [None]),
    ],
    ids=[
        "first string",
        "first null",
        "second string",
        "second null",
    ],
)
def test_should_raise_if_columns_are_not_numeric(values_1: list, values_2: list) -> None:
    column1 = Column("col1", values_1)
    column2 = Column("col2", values_2)
    with pytest.raises(ColumnTypeError):
        column1.correlation_with(column2)


def test_should_raise_if_row_counts_differ() -> None:
    column1 = Column("col1", [1, 2, 3, 4])
    column2 = Column("col2", [2])
    with pytest.raises(LengthMismatchError):
        column1.correlation_with(column2)


@pytest.mark.parametrize(
    ("values_1", "values_2"),
    [
        ([None, 2], [1, 2]),
        ([1, 2], [1, None]),
    ],
    ids=[
        "first missing",
        "second missing",
    ],
)
def test_should_raise_if_columns_have_missing_values(values_1: list, values_2: list) -> None:
    column1 = Column("col1", values_1)
    column2 = Column("col2", values_2)
    with pytest.raises(MissingValuesError):
        column1.correlation_with(column2)
