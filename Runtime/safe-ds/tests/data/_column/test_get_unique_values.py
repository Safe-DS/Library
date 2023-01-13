import typing

import pytest
from safe_ds.data import Column


@pytest.mark.parametrize(
    "values, unique_values",
    [([1, 1, 2, 3], [1, 2, 3]), (["a", "b", "b", "c"], ["a", "b", "c"]), ([], [])],
)
def test_get_unique_values(
    values: list[typing.Any], unique_values: list[typing.Any]
) -> None:
    column: Column = Column(values, "")
    extracted_unique_values: list[typing.Any] = column.get_unique_values()

    assert extracted_unique_values == unique_values
