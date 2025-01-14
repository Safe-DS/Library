import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], 1),
        ([1, 2, 3], 0),
        ([1, 2, 3, None], 1 / 4),
        ([None, None, None], 1),
    ],
    ids=[
        "empty",
        "no missing values",
        "some missing values",
        "only missing values",
    ],
)
def test_should_return_the_missing_value_ratio(values: list, expected: float) -> None:
    column = Column("col1", values)
    assert column.missing_value_ratio() == expected
