import math

import pytest

from tests.helpers import assert_cell_operation_works


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
        "-1",
        "0",
        "1",
        "square of 1.5",
        "square of 2",
        "None",
    ],
)
def test_should_return_square_root(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.sqrt(), expected)
