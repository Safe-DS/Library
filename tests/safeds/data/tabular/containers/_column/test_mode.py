import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], []),
        ([None, None, None], []),
        ([1, 2, 2], [2]),
        (["a", "a", "b"], ["a"]),
        ([1, 2, 2, None], [2]),
        ([1, 2], [1, 2]),
    ],
    ids=[
        "empty",
        "all missing values",
        "numeric",
        "non-numeric",
        "some missing values",
        "multiple values with same frequency",
    ],
)
def test_should_return_the_mode_value(values: list, expected: list) -> None:
    column = Column("A", values)
    assert column.mode() == expected
