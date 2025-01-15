from datetime import date

import pytest

from safeds._typing import _ConvertibleToIntCell
from safeds.data.tabular.containers import Cell
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("year", "month", "day", "expected"),
    [
        (1, 2, 3, date(1, 2, 3)),
        (Cell.constant(1), Cell.constant(2), Cell.constant(3), date(1, 2, 3)),
        # invalid year
        (None, 2, 3, None),
        # invalid month
        (1, None, 3, None),
        (1, 0, 3, None),
        (1, 13, 3, None),
        # invalid day
        (1, 2, None, None),
        (1, 2, 0, None),
        (1, 2, 32, None),
    ],
    ids=[
        "int components",
        "cell components",
        "year is None",
        "month is None",
        "month is too low",
        "month is too high",
        "day is None",
        "day is too low",
        "day is too high",
    ],
)
def test_should_return_date(
    year: _ConvertibleToIntCell,
    month: _ConvertibleToIntCell,
    day: _ConvertibleToIntCell,
    expected: date,
) -> None:
    assert_cell_operation_works(None, lambda _: Cell.date(year, month, day), expected)
