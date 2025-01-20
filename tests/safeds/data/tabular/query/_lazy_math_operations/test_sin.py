import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (math.pi / 2, 1),
        (math.pi, 0),
        (3 * math.pi / 2, -1),
        (2 * math.pi, 0),
        (None, None),
    ],
    ids=[
        "0 deg",
        "90 deg",
        "180 deg",
        "270 deg",
        "360 deg",
        "None",
    ],
)
def test_should_return_sine(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.sin(), expected, type_if_none=ColumnType.float64())
