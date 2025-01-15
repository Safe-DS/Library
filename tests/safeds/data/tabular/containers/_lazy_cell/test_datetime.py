from datetime import datetime

import pytest
from helpers import assert_cell_operation_works

from safeds._typing import _ConvertibleToIntCell
from safeds.data.tabular.containers import Cell


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "microsecond", "expected"),
    [
        (1, 2, 3, 4, 5, 6, 7, datetime(1, 2, 3, 4, 5, 6, 7)),  # noqa: DTZ001
        (
            Cell.constant(1),
            Cell.constant(2),
            Cell.constant(3),
            Cell.constant(4),
            Cell.constant(5),
            Cell.constant(6),
            Cell.constant(7),
            datetime(1, 2, 3, 4, 5, 6, 7),  # noqa: DTZ001
        ),
        # invalid year
        (None, 2, 3, 4, 5, 6, 7, None),
        # invalid month
        (1, None, 3, 4, 5, 6, 7, None),
        (1, 0, 3, 4, 5, 6, 7, None),
        (1, 13, 3, 4, 5, 6, 7, None),
        # invalid day
        (1, 2, None, 4, 5, 6, 7, None),
        (1, 2, 0, 4, 5, 6, 7, None),
        (1, 2, 32, 4, 5, 6, 7, None),
        # invalid hour
        (1, 2, 3, None, 5, 6, 7, None),
        (1, 2, 3, -1, 5, 6, 7, None),
        (1, 2, 3, 24, 5, 6, 7, None),
        # invalid minute
        (1, 2, 3, 4, None, 6, 7, None),
        (1, 2, 3, 4, -1, 6, 7, None),
        (1, 2, 3, 4, 60, 6, 7, None),
        # invalid second
        (1, 2, 3, 4, 5, None, 7, None),
        (1, 2, 3, 4, 5, -1, 7, None),
        (1, 2, 3, 4, 5, 60, 7, None),
        # invalid microsecond
        (1, 2, 3, 4, 5, 6, None, None),
        (1, 2, 3, 4, 5, 6, -1, None),
        (1, 2, 3, 4, 5, 6, 1_000_000, None),
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
        "hour is None",
        "hour is too low",
        "hour is too high",
        "minute is None",
        "minute is too low",
        "minute is too high",
        "second is None",
        "second is too low",
        "second is too high",
        "microsecond is None",
        "microsecond is too low",
        "microsecond is too high",
    ],
)
def test_should_return_constant_value(
    year: _ConvertibleToIntCell,
    month: _ConvertibleToIntCell,
    day: _ConvertibleToIntCell,
    hour: _ConvertibleToIntCell,
    minute: _ConvertibleToIntCell,
    second: _ConvertibleToIntCell,
    microsecond: _ConvertibleToIntCell,
    expected: datetime,
) -> None:
    assert_cell_operation_works(
        None,
        lambda _: Cell.datetime(
            year,
            month,
            day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond,
        ),
        expected,
    )
