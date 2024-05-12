from typing import Any

import pytest
from safeds.data.tabular.containers import Column


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
        (slice(0, 0), Column("a", [])),
        (slice(0, 1), Column("a", [0])),
        (slice(2, 3), Column("a", [2])),
        (slice(0, 3), Column("a", [0, 1, 2])),
        (slice(0, 3, 2), Column("a", [0, 2])),
        (slice(100, 101), Column("a", [])),
        (slice(2, 10), Column("a", [2])),
    ],
    ids=[
        "empty",
        "first item",
        "last item",
        "all items",
        "every other item",
        "large start index",
        "large end index",
    ],
)
def test_should_get_the_items_at_slice(index: slice, expected: Column) -> None:
    assert Column("a", [0, 1, 2])[index] == expected


@pytest.mark.parametrize(
    "index",
    [
        2,
        slice(-1, 2),
        slice(0, -1),
        slice(0, 3, -1),
    ],
    ids=[
        "index out of bounds",
        "slice with negative start",
        "slice with negative end",
        "slice with negative step",
    ],
)
def test_should_raise_if_index_is_out_of_bounds(index: int | slice) -> None:
    column = Column("a", [0, 1])

    with pytest.raises(IndexError):
        _result = column[index]
