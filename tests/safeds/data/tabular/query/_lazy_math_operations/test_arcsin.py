import math

import pytest
from helpers import assert_cell_operation_works

from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1.0, -math.pi / 2),
        (0, 0),
        (1, math.pi / 2),
        (None, None),
    ],
    ids=[
        "-90 deg",
        "0 deg",
        "90 deg",
        "None",
    ],
)
def test_should_return_inverse_sine(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.arcsin(), expected, type_if_none=ColumnType.float64())
