import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import NonNumericColumnError


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], None),
        ([1, 2, 3], 2),
        ([1, 2, 3, None], 2),
        (["a", "b", "c"], None),
        ([None, None, None], None),
    ],
    ids=[
        "empty",
        "no missing values",
        "some missing values",
        "non-numeric",
        "all missing values",
    ],
)
def test_should_return_the_mean_value(values: list, expected: int) -> None:
    column = Column("A", values)
    assert column.mean() == expected
