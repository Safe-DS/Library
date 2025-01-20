import math

import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "base", "expected"),
    [
        # base e
        (0, math.e, -math.inf),
        (1, math.e, 0),
        (math.e, math.e, 1),
        # base 10
        (0, 10, -math.inf),
        (1, 10, 0),
        (10, 10, 1),
        (100, 10, 2),
        # None
        (None, 10, None),
    ],
    ids=[
        # base e
        "base e - 0",
        "base e - 1",
        "base e - e",
        # base 10
        "base 10 - 0",
        "base 10 - 1",
        "base 10 - 10",
        "base 10 - 100",
        # None
        "None",
    ],
)
def test_should_return_logarithm_to_given_base(value: float | None, base: int, expected: float | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.math.log(base), expected, type_if_none=ColumnType.float64())


@pytest.mark.parametrize(
    "base",
    [
        -1,
        0,
        1,
    ],
    ids=[
        "negative",
        "zero",
        "one",
    ],
)
def test_should_raise_if_base_is_out_of_bounds(base: int) -> None:
    column = Column("a", [1])
    with pytest.raises(ValueError, match="base"):
        column.transform(lambda cell: cell.math.log(base))
