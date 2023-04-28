import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import NonNumericColumnError


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3], 6),
        ([1, 2, 3, 4], 10),
        ([1, 2, 3, None], 6),
    ],
    ids=[
        "odd number of values",
        "even number of values",
        "some missing values",
    ],
)
def test_should_return_sum_of_values(values: list, expected: int) -> None:
    column = Column("A", values)
    assert column.sum() == expected


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
    with pytest.raises(NonNumericColumnError):
        column.sum()
