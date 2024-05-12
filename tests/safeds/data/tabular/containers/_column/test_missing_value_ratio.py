import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3], 0),
        ([1, 2, 3, None], 1 / 4),
        ([None, None, None], 1),
    ],
    ids=[
        "no missing values",
        "some missing values",
        "only missing values",
    ],
)
def test_should_count_missing_values(values: list, expected: float) -> None:
    column = Column("count", values)
    assert column.missing_value_ratio() == expected
