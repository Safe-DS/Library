from typing import Any

import pytest
from safeds.data.tabular.containers import Column
from safeds.exceptions import IndexOutOfBoundsError


@pytest.mark.parametrize(
    ("column", "index", "expected"),
    [
        (Column("a", [0, 1]), 0, 0),
        (Column("a", [0, 1]), 1, 1),
    ],
    ids=["first item", "second item"],
)
def test_should_get_the_item_at_index(column: Column, index: int, expected: Any) -> None:
    assert column[index] == expected


@pytest.mark.parametrize(
    ("column", "index", "expected"),
    [
        (Column("a", [0, 1, 2]), slice(0, 1), Column("a", [0])),
        (Column("a", [0, 1, 2]), slice(2, 3), Column("a", [2])),
        (Column("a", [0, 1, 2]), slice(0, 3), Column("a", [0, 1, 2])),
        (Column("a", [0, 1, 2]), slice(0, 3, 2), Column("a", [0, 2])),
    ],
    ids=["first item", "last item", "all items", "every other item"],
)
def test_should_get_the_items_at_slice(column: Column, index: slice, expected: Column) -> None:
    assert column[index] == expected


@pytest.mark.parametrize(
    "index",
    [-1, 2, slice(-1, 2), slice(0, 4), slice(-1, 4)],
    ids=[
        "negative",
        "out of bounds",
        "slice with negative start",
        "slice with out of bounds end",
        "slice with negative start and out of bounds end",
    ],
)
def test_should_raise_if_index_is_out_of_bounds(index: int | slice) -> None:
    column = Column("a", [0, "1"])

    with pytest.raises(IndexOutOfBoundsError):
        # noinspection PyStatementEffect
        column[index]
