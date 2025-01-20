import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, -math.inf),
        (1, 0),
        (math.e, 1),
        (None, None),
    ],
    ids=[
        "0",
        "1",
        "e",
        "None",
    ],
)
def test_should_return_natural_logarithm(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.ln(), expected, type_if_none=ColumnType.float64())
