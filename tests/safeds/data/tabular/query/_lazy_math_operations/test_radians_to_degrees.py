import math

import pytest
from helpers import assert_cell_operation_works

from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (math.pi / 2, 90),
        (math.pi, 180),
        (3 * math.pi / 2, 270),
        (2 * math.pi, 360),
        (4 * math.pi, 720),
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
def test_should_convert_radians_to_degrees(value: int | None, expected: int | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.math.radians_to_degrees(),
        expected,
        type_if_none=ColumnType.float64(),
    )
