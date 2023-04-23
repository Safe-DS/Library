import pytest
from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], False),
        ([1, 2, 3], False),
        ([1, 2, 3, None], True),
        ([None, None, None], True),
    ],
    ids=[
        "empty",
        "no missing values",
        "some missing values",
        "all missing values",
    ],
)
def test_should_return_whether_the_column_has_missing_values(values: list, expected: bool) -> None:
    column = Column("A", values)
    assert column.has_missing_values() == expected
