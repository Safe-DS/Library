import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], 0),
        ([1, 2, 3], 0),
        ([1, 2, 3, None], 1),
        ([None, None, None], 3),
    ],
    ids=[
        "empty",
        "no missing values",
        "some missing values",
        "only missing values",
    ],
)
def test_should_count_missing_values(values: list, expected: float) -> None:
    column = Column("col1", values)
    assert column.missing_value_count() == expected
