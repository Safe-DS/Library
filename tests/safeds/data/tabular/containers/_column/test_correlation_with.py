import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import ColumnLengthMismatchError, NonNumericColumnError


@pytest.mark.parametrize(
    ("column1", "column2", "expected"),
    [
        (Column("A", [0, 1, 2]), Column("B", [0, 1, 2]), 1.0),
        (Column("A", [0, 1, 2]), Column("B", [0, -1, -2]), -1.0),
    ],
    ids=["positive correlation", "negative correlation"],
)
def test_should_return_correlation_between_two_columns(column1: Column, column2: Column, expected: float) -> None:
    assert column1.correlation_with(column2) == expected


@pytest.mark.parametrize(
    ("column1", "column2"),
    [
        (Column("A", [1]), Column("B", ["b"])),
        (Column("A", ["a"]), Column("B", [2.0])),
        (Column("A", ["a"]), Column("B", ["a"])),
    ],
    ids=[
        "first column is not numeric",
        "second column is not numeric",
        "both columns are not numeric",
    ],
)
def test_should_raise_if_columns_are_not_numeric(column1: Column, column2: Column) -> None:
    with pytest.raises(NonNumericColumnError):
        column1.correlation_with(column2)


def test_should_raise_if_column_lengths_differ() -> None:
    column1 = Column("A", [1, 2, 3, 4])
    column2 = Column("B", [2])
    with pytest.raises(ColumnLengthMismatchError):
        column1.correlation_with(column2)
