import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import ColumnLengthMismatchError, MissingValuesColumnError, NonNumericColumnError


@pytest.mark.parametrize(
    ("values1", "values2", "expected"),
    [
        ([0, 1, 2], [0, 1, 2], 1.0),
        ([0, 1, 2], [0, -1, -2], -1.0),
    ],
    ids=[
        "positive correlation",
        "negative correlation",
    ],
)
def test_should_return_correlation_between_two_columns(values1: list, values2: list, expected: float) -> None:
    column1 = Column("A", values1)
    column2 = Column("B", values2)
    assert column1.correlation_with(column2) == expected


@pytest.mark.parametrize(
    ("values1", "values2"),
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
def test_should_raise_if_columns_are_not_numeric(values1: list, values2: list) -> None:
    column1 = Column("A", values1)
    column2 = Column("B", values2)
    with pytest.raises(NonNumericColumnError):
        column1.correlation_with(column2)


def test_should_raise_if_column_lengths_differ() -> None:
    column1 = Column("A", [1, 2, 3, 4])
    column2 = Column("B", [2])
    with pytest.raises(ColumnLengthMismatchError):
        column1.correlation_with(column2)


@pytest.mark.parametrize(
    ("values1", "values2"),
    [
        ([None, 2], [1, 2]),
        ([1, 2], [1, None]),
    ],
    ids=[
        "first missing",
        "second missing",
    ],
)
def test_should_raise_if_columns_have_missing_values(values1: list, values2: list) -> None:
    column1 = Column("A", values1)
    column2 = Column("B", values2)
    with pytest.raises(MissingValuesColumnError):
        column1.correlation_with(column2)
