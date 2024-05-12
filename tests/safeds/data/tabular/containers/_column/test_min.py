import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], None),
        ([1, 2, 3], 1),
        ([1, 2, 3, None], 1),
        ([None, None], None),

    ],
    ids=[
        "empty",
        "no missing values",
        "some missing values",
        "only missing values",
    ],
)
def test_should_return_max_value(values: list, expected: int) -> None:
    column = Column("col", values)
    assert column.min() == expected
