import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], []),
        ([None, 2, 2, None], [2]),
        ([None, None, None], []),
        ([1, 2, 2], [2]),
        (["a", "a", "b"], ["a"]),
        ([1, 2], [1, 2]),
    ],
    ids=[
        "empty",
        "some missing values",
        "all missing values",
        "numeric",
        "non-numeric",
        "multiple values with same frequency",
    ],
)
def test_should_return_mode_values_ignoring_missing_values(values: list, expected: list) -> None:
    column = Column("col", values)
    assert column.mode() == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], []),
        ([None, 2, 2, None], [None, 2]),
        ([None, None, None], [None]),
        ([1, 2, 2], [2]),
        (["a", "a", "b"], ["a"]),
        ([1, 2], [1, 2]),
    ],
    ids=[
        "empty",
        "some missing values",
        "all missing values",
        "numeric",
        "non-numeric",
        "multiple values with same frequency",
    ],
)
def test_should_return_mode_values_including_missing_values_if_requested(values: list, expected: list) -> None:
    column = Column("col", values)
    assert column.mode(ignore_missing_values=False) == expected
