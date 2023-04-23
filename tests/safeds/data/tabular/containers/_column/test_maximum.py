import pytest
from safeds.data.tabular.containers import Table, Column
from safeds.data.tabular.exceptions import NonNumericColumnError


@pytest.mark.parametrize(
    ("values", "expected"),
    [

        ([1, 2, 3], 3),
        ([1, 2, 3, None], 3),
    ],
    ids=[
        "no missing values",
        "some missing values",
    ],
)
def test_should_return_the_maximum_value(values: list, expected: int) -> None:
    column = Column("A", values)
    assert column.maximum() == expected


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
        column.maximum()
