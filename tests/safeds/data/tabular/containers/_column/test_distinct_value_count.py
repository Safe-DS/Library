import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "ignore_missing_values", "expected"),
    [
        ([], True, 0),
        ([1, 2, 3], True, 3),
        ([1, 2, 1], True, 2),
        ([1, 2, 3, None], True, 3),
        ([1, 2, 3, None], False, 4),
    ],
    ids=[
        "empty",
        "no duplicates",
        "some duplicate",
        "with missing values (ignored)",
        "with missing values (not ignored)",
    ],
)
def test_should_return_number_of_distinct_values(values: list, ignore_missing_values: bool, expected: int) -> None:
    column = Column("col", values)
    assert column.distinct_value_count(ignore_missing_values=ignore_missing_values) == expected
