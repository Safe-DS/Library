import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "ignore_missing_values", "expected"),
    [
        ([], True, []),
        ([1, 2, 2], True, [2]),
        (["a", "a", "b"], True, ["a"]),
        ([1, 2, None], True, [1, 2]),
        ([1, 2, None], False, [None, 1, 2]),
        ([None, 2, None], True, [2]),
        ([None, 2, None], False, [None]),
        ([None, None, None], True, []),
        ([None, None, None], False, [None]),
    ],
    ids=[
        "empty",
        "numeric",
        "non-numeric",
        "multiple most frequent values (missing values ignored)",
        "multiple most frequent values (missing values not ignored)",
        "missing values are most frequent (missing values ignored)",
        "missing values are most frequent (missing values not ignored)",
        "only missing values (missing values ignored)",
        "only missing values (missing values not ignored)",
    ],
)
def test_should_return_most_frequent_values(values: list, ignore_missing_values: bool, expected: list) -> None:
    column = Column("col1", values)
    assert column.mode(ignore_missing_values=ignore_missing_values) == expected
