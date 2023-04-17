import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3], False),
        ([1, 2, 3, None], True),
        ([None, None, None], True),
        ([], False),
    ],
)
def test_has_missing_values(values: list, expected: bool) -> None:
    if len(values) == 0:
        column = Column("A", values)
    else:
        column = Column("A", values)
    assert column.has_missing_values() == expected
