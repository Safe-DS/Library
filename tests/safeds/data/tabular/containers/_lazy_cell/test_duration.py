from datetime import timedelta

import pytest
from helpers import assert_cell_operation_works

from safeds._typing import _ConvertibleToIntCell
from safeds.data.tabular.containers import Cell


@pytest.mark.parametrize(
    ("weeks", "days", "hours", "minutes", "seconds", "milliseconds", "microseconds", "expected"),
    [
        (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            timedelta(weeks=1, days=2, hours=3, minutes=4, seconds=5, milliseconds=6, microseconds=7),
        ),
        (
            -1,
            -2,
            -3,
            -4,
            -5,
            -6,
            -7,
            timedelta(weeks=-1, days=-2, hours=-3, minutes=-4, seconds=-5, milliseconds=-6, microseconds=-7),
        ),
        (
            Cell.constant(1),
            Cell.constant(2),
            Cell.constant(3),
            Cell.constant(4),
            Cell.constant(5),
            Cell.constant(6),
            Cell.constant(7),
            timedelta(weeks=1, days=2, hours=3, minutes=4, seconds=5, milliseconds=6, microseconds=7),
        ),
        (None, 2, 3, 4, 5, 6, 7, None),
        (1, None, 3, 4, 5, 6, 7, None),
        (1, 2, None, 4, 5, 6, 7, None),
        (1, 2, 3, None, 5, 6, 7, None),
        (1, 2, 3, 4, None, 6, 7, None),
        (1, 2, 3, 4, 5, None, 7, None),
        (1, 2, 3, 4, 5, 6, None, None),
    ],
    ids=[
        "positive int components",
        "negative int components",
        "cell components",
        "weeks is None",
        "days is None",
        "hours is None",
        "minutes is None",
        "seconds is None",
        "microseconds is None",
        "milliseconds is None",
    ],
)
def test_should_return_duration(
    weeks: _ConvertibleToIntCell,
    days: _ConvertibleToIntCell,
    hours: _ConvertibleToIntCell,
    minutes: _ConvertibleToIntCell,
    seconds: _ConvertibleToIntCell,
    milliseconds: _ConvertibleToIntCell,
    microseconds: _ConvertibleToIntCell,
    expected: timedelta,
) -> None:
    assert_cell_operation_works(
        None,
        lambda _: Cell.duration(
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            milliseconds=milliseconds,
            microseconds=microseconds,
        ),
        expected,
    )
