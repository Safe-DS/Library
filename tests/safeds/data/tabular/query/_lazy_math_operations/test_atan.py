import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1.0, -math.pi / 4),
        (0, 0),
        (1, math.pi / 4),
        (None, None),
    ],
    ids=[
        "-45 deg",
        "0 deg",
        "45 deg",
        "None",
    ],
)
def test_should_return_inverse_tangent(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.atan(), expected, type_if_none=ColumnType.float64())
