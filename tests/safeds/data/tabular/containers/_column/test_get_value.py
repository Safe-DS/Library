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
    assert Column("a", [0, 1]).get_value(index) == expected


@pytest.mark.parametrize(
    "index",
    [-3, 2],
    ids=["too low", "too high"],
)
def test_should_raise_if_index_is_out_of_bounds(index: int) -> None:
    column = Column("a", [0, 1])
    with pytest.raises(IndexOutOfBoundsError):
        column.get_value(index)
