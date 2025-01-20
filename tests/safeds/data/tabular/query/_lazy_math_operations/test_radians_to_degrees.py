import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (math.pi / 8, 22.5),
        (math.pi / 2, 90),
        (math.pi, 180),
        (3 * math.pi / 2, 270),
        (2 * math.pi, 360),
        (4 * math.pi, 720),
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
def test_should_convert_radians_to_degrees(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.math.radians_to_degrees(),
        expected,
        type_if_none=ColumnType.float64(),
    )
