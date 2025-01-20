import math

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, -math.inf),
        (1, 0),
        (10, 1),
        (100, 2),
        (None, None),
    ],
    ids=[
        "0",
        "1",
        "10",
        "100",
        "None",
    ],
)
def test_should_return_common_logarithm(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.log10(), expected, type_if_none=ColumnType.float64())
