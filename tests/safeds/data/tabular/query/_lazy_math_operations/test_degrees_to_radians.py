import math

import pytest
from helpers import assert_cell_operation_works

from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (22.5, math.pi / 8),
        (90, math.pi / 2),
        (180, math.pi),
        (270, 3 * math.pi / 2),
        (360, 2 * math.pi),
        (720, 4 * math.pi),
        (None, None),
    ],
    ids=[
        "0 deg",
        "22.5 deg",
        "90 deg",
        "180 deg",
        "270 deg",
        "360 deg",
        "720 deg",
        "None",
    ],
)
def test_should_convert_degrees_to_radians(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.math.degrees_to_radians(),
        expected,
        type_if_none=ColumnType.float64(),
    )
