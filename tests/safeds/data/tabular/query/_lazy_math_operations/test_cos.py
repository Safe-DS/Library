import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 1),
        (math.pi / 2, 0),
        (math.pi, -1),
        (3 * math.pi / 2, 0),
        (2 * math.pi, 1),
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
def test_should_return_cosine(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.cos(), expected, type_if_none=ColumnType.float64())
