from typing import Any

import pytest

from safeds.data.tabular.containers import Column
from safeds.exceptions import IndexOutOfBoundsError


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, 0),
        (1, 1),
        (-2, 0),
        (-1, 1),
    ],
    ids=[
        "first item (positive index)",
        "last item (positive index)",
        "first item (negative index)",
        "last item (negative index)",
    ],
)
def test_should_get_the_item_at_index(index: int, expected: Any) -> None:
    assert Column("a", [0, 1])[index] == expected


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        # empty
        (slice(0, 0), []),
        # first item
        (slice(0, 1), [0]),
        # last item
        (slice(2, 3), [2]),
        # all items
        (slice(0, 3), [0, 1, 2]),
        # every other item
        (slice(0, 3, 2), [0, 2]),
        # negative start index
        (slice(-1, None), [2]),
        # negative end index
        (slice(None, -1), [0, 1]),
        # negative step
        (slice(None, None, -1), [2, 1, 0]),
        # high start index (step=1)
        (slice(100, None), []),
        # high start index (step=-1)
        (slice(100, None, -1), [2, 1, 0]),
        # high end index (step=1)
        (slice(None, 100), [0, 1, 2]),
        # high end index (step=-1)
        (slice(None, 100, -1), []),
        # low start index (step=1)
        (slice(-100, None), [0, 1, 2]),
        # low start index (step=-1)
        (slice(-100, None, -1), []),
        # low end index (step=1)
        (slice(None, -100), []),
        # low end index (step=-1)
        (slice(None, -100, -1), [2, 1, 0]),
    ],
    ids=[
        "empty",
        "first item",
        "last item",
        "all items",
        "every other item",
        "negative start index",
        "negative end index",
        "negative step",
        "high start index (step=1)",
        "high start index (step=-1)",
        "high end index (step=1)",
        "high end index (step=-1)",
        "low start index (step=1)",
        "low start index (step=-1)",
        "low end index (step=1)",
        "low end index (step=-1)",
    ],
)
def test_should_get_the_items_at_slice(index: slice, expected: list) -> None:
    assert list(Column("a", [0, 1, 2])[index]) == expected


@pytest.mark.parametrize(
    "index",
    [-3, 2],
    ids=["too low", "too high"],
)
def test_should_raise_if_index_is_out_of_bounds(index: int) -> None:
    column = Column("a", [0, 1])
    with pytest.raises(IndexOutOfBoundsError):
        column.get_value(index)
