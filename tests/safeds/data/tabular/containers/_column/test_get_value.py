from typing import Any

import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import IndexOutOfBoundsError


@pytest.mark.parametrize(
    ("column", "index", "expected"),
    [
        (Column("a", [0, 1]), 0, 0),
        (Column("a", [0, 1]), 1, 1),
    ],
    ids=["first item", "second item"],
)
def test_should_get_the_item_at_index(column: Column, index: int, expected: Any) -> None:
    assert column.get_value(index) == expected


@pytest.mark.parametrize(
    "index",
    [-1, 2],
    ids=[
        "negative",
        "out of bounds",
    ],
)
def test_should_raise_if_index_is_out_of_bounds(index: int | slice) -> None:
    column = Column("a", [0, "1"])

    with pytest.raises(IndexOutOfBoundsError):
        column.get_value(index)
