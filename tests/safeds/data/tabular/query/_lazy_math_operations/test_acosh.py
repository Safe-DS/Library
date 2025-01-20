import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works

E = math.e
PHI = (1 + math.sqrt(5)) / 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 0),
        (0.5 * (E + 1 / E), 1),
        (0.5 * math.sqrt(5), math.log(PHI)),
        (None, None),
    ],
    ids=[
        "0",
        "1",
        "ln of golden ratio",
        "None",
    ],
)
def test_should_return_inverse_hyperbolic_cosine(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.acosh(), expected, type_if_none=ColumnType.float64())
