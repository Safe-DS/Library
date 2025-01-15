from datetime import date

import pytest
from helpers import assert_cell_operation_works

from safeds._typing import _ConvertibleToIntCell
from safeds.data.tabular.containers import Cell


@pytest.mark.parametrize(
    ("year", "month", "day", "expected"),
    [
        (2025, 1, 1, date(2025, 1, 1)),
        (Cell.constant(2025), Cell.constant(1), Cell.constant(1), date(2025, 1, 1)),
        (None, 1, 1, None),
        (2025, None, 1, None),
        (2025, 0, 1, None),
        (2025, 13, 1, None),
        (2025, 1, None, None),
        (2025, 1, 0, None),
        (2025, 1, 32, None),
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
def test_should_return_constant_value(
    year: _ConvertibleToIntCell,
    month: _ConvertibleToIntCell,
    day: _ConvertibleToIntCell,
    expected: date,
) -> None:
    assert_cell_operation_works(None, lambda _: Cell.date(year, month, day), expected)
