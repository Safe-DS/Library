import math

import pytest
from helpers import assert_cell_operation_works

from safeds.data.tabular.typing import ColumnType

E = math.e
PHI = (1 + math.sqrt(5)) / 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (-(E - 1 / E) / (E + 1 / E), -1),
        ((E - 1 / E) / (E + 1 / E), 1),
        (1 / math.sqrt(5), math.log(PHI)),
        (None, None),
    ],
    ids=[
        "0",
        "-1",
        "1",
        "ln of golden ratio",
        "None",
    ],
)
def test_should_return_inverse_hyperbolic_tangent(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.artanh(), expected, type_if_none=ColumnType.float64())
