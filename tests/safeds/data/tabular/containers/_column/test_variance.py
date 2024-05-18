import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import ColumnTypeError


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3], 1),
        ([1, 2, 3, None], 1),
    ],
    ids=[
        "no missing values",
        "some missing values",
    ],
)
def test_should_return_variance(values: list, expected: int) -> None:
    column = Column("A", values)
    assert column.variance() == expected


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
        column.variance()
