import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], 0),
        ([1, 2, 3], 3),
        ([1, 2, 1], 2),
        ([1, 2, 3, None], 3),
    ],
    ids=[
        "empty",
        "no duplicates",
        "some duplicate",
        "with missing values",
    ],
)
def test_should_return_number_of_distinct_values_ignoring_missing_values(values: list, expected: int) -> None:
    column = Column("A", values)
    assert column.distinct_value_count() == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3, None], 4),
        ([1, 2, None, None], 3),
    ],
    ids=[
        "with one missing value",
        "with several missing values",
    ],
)
def test_should_return_number_of_distinct_values_including_missing_values_if_requested(
    values: list,
    expected: int,
) -> None:
    column = Column("A", values)
    assert column.distinct_value_count(ignore_missing_values=False) == expected
