import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 2, 3], 0), ([1, 2, 3, None], 1 / 4), ([None, None, None], 1)],
)
def test_missing_value_ratio(values: list, expected: float) -> None:
    column = Column("A", values)
    result = column.missing_value_ratio()
    assert result == expected


def test_missing_value_ratio_empty() -> None:
    column = Column("A", [])
    with pytest.raises(ColumnSizeError):
        column.missing_value_ratio()
