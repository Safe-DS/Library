import math

import pytest
from helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, math.nan),
        (0, 0),
        (1, 1),
        (2.25, 1.5),
        (4, 2),
        (None, None),
    ],
    ids=[
        "negative",
        "zero",
        "one",
        "square of flot",
        "square of int",
        "None",
    ],
)
def test_should_return_square_root(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.sqrt(), expected)
