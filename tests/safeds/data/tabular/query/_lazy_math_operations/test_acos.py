import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, 0),
        (0, math.pi / 2),
        (-1, math.pi),
        (None, None),
    ],
    ids=[
        "0 deg",
        "90 deg",
        "180 deg",
        "None",
    ],
)
def test_should_return_inverse_cosine(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.acos(), expected, type_if_none=ColumnType.float64())
