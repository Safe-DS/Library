import pytest
from helpers import assert_cell_operation_works

from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-2.5, -1),
        (-1, -1),
        (-0, -0),
        (0, 0),
        (1, 1),
        (2.5, 1),
        (None, None),
    ],
    ids=[
        "-2.5",
        "-1",
        "-0",
        "0",
        "1",
        "2.5",
        "None",
    ],
)
def test_should_return_sign(value: float | None, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.sign(), expected, type_if_none=ColumnType.float64())
