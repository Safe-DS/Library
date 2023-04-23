import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 2, 3], 0), ([1, 2, 3, None], 1 / 4), ([None, None, None], 1)],
    ids=["no missing values", "some missing values", "all missing values"],
)
def test_should_return_ratio_of_null_values_to_number_of_elements(values: list, expected: float) -> None:
    column = Column("A", values)
    assert column.missing_value_ratio() == expected


def test_should_raise_if_column_is_empty() -> None:
    column = Column("A", [])
    with pytest.raises(ColumnSizeError):
        column.missing_value_ratio()
