import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 2, 3], 0), ([1, 2, 3, None], 1), ([None, None, None], 3)],
    ids=["no missing values", "some missing values", "all missing values"],
)
def test_should_count_missing_values(values: list, expected: float) -> None:
    column = Column("A", values)
    assert column._count_missing_values() == expected
