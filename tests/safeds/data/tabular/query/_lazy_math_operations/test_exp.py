import math

import pytest
from helpers import assert_cell_operation_works

from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 1),
        (1.0, math.e),
        (None, None),
    ],
    ids=[
        "0",
        "1",
        "None",
    ],
)
def test_should_return_exponential(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.exp(), expected, type_if_none=ColumnType.float64())
