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
def test_should_return_median_value(values: list, expected: int) -> None:
    column = Column("A", values)
    assert column.median() == expected


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["a", "b", "c"],
        [None, None, None],
    ],
    ids=[
        "empty",
        "non-numeric",
        "all missing values",
    ],
)
def test_should_raise_if_column_is_not_numeric(values: list) -> None:
    column = Column("A", values)
    with pytest.raises(ColumnTypeError):
        column.median()
