import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works

E = math.e
PHI = (1 + math.sqrt(5)) / 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (-1, -(E - 1 / E) / (E + 1 / E)),
        (1, (E - 1 / E) / (E + 1 / E)),
        (math.log(PHI), 1 / math.sqrt(5)),
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
def test_should_return_hyperbolic_tangent(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.tanh(), expected, type_if_none=ColumnType.float64())
