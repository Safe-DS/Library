import pytest

from safeds.data.tabular.containers import Column
from safeds.exceptions import ColumnTypeError


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3], 2),
        ([1, 2, 3, 4], 2.5),
        ([1, 2, 3, None], 2),
    ],
    ids=[
        "odd number of values",
        "even number of values",
        "some missing values",
    ],
)
def test_should_return_median(values: list, expected: int) -> None:
    column = Column("col1", values)
    assert column.median() == expected


@pytest.mark.parametrize(
    "values",
    [
        [],
        [None],
        ["a"],
    ],
    ids=[
        "empty",
        "all missing values",
        "non-numeric",
    ],
)
def test_should_raise_if_column_is_not_numeric(values: list) -> None:
    column = Column("col1", values)
    with pytest.raises(ColumnTypeError):
        column.median()
