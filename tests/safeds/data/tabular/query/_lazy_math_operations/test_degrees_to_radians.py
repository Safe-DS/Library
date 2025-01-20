import math

import pytest
from helpers import assert_cell_operation_works

from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (90, math.pi / 2),
        (180, math.pi),
        (270, 3 * math.pi / 2),
        (360, 2 * math.pi),
        (720, 4 * math.pi),
        (None, None),
    ],
    ids=[
        "0°",
        "90°",
        "180°",
        "270°",
        "360°",
        "720°",
        "None",
    ],
)
def test_should_convert_degrees_to_radians(value: int | None, expected: int | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.math.degrees_to_radians(),
        expected,
        type_if_none=ColumnType.float64(),
    )
