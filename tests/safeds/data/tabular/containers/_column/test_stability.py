from typing import Any

import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3, 4], 1 / 4),
        ([1, 2, 3, 4, None], 1 / 4),
        (["b", "a", "abc", "abc", "abc"], 3 / 5),
        ([1, 1, 3, "abc", None], 2 / 4),
    ],
    ids=[
        "numeric",
        "numeric with missing values",
        "non-numeric",
        "mixed with missing values",
    ],
)
def test_should_return_stability_of_column(values: list[Any], expected: float) -> None:
    column = Column("A", values)
    assert column.stability() == expected


def test_should_raise_column_size_error_if_column_is_empty() -> None:
    column: Column[Any] = Column("A", [])
    with pytest.raises(ColumnSizeError, match="Expected a column of size > 0 but got column of size 0."):
        column.stability()


def test_should_raise_value_error_if_column_contains_only_none() -> None:
    column: Column[Any] = Column("A", [None, None])
    with pytest.raises(ValueError, match="Stability is not definded for a column with only null values."):
        column.stability()
