from datetime import UTC, datetime

import pytest

from safeds._typing import _ConvertibleToIntCell
from safeds.data.tabular.containers import Cell, Column
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "microsecond", "time_zone", "expected"),
    [
        (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            None,
            datetime(1, 2, 3, 4, 5, 6, 7),  # noqa: DTZ001
        ),
        (
            Cell.constant(1),
            Cell.constant(2),
            Cell.constant(3),
            Cell.constant(4),
            Cell.constant(5),
            Cell.constant(6),
            Cell.constant(7),
            None,
            datetime(1, 2, 3, 4, 5, 6, 7),  # noqa: DTZ001
        ),
        # invalid year
        (None, 2, 3, 4, 5, 6, 7, None, None),
        # invalid month
        (1, None, 3, 4, 5, 6, 7, None, None),
        (1, 0, 3, 4, 5, 6, 7, None, None),
        (1, 13, 3, 4, 5, 6, 7, None, None),
        # invalid day
        (1, 2, None, 4, 5, 6, 7, None, None),
        (1, 2, 0, 4, 5, 6, 7, None, None),
        (1, 2, 32, 4, 5, 6, 7, None, None),
        # invalid hour
        (1, 2, 3, None, 5, 6, 7, None, None),
        (1, 2, 3, -1, 5, 6, 7, None, None),
        (1, 2, 3, 24, 5, 6, 7, None, None),
        # invalid minute
        (1, 2, 3, 4, None, 6, 7, None, None),
        (1, 2, 3, 4, -1, 6, 7, None, None),
        (1, 2, 3, 4, 60, 6, 7, None, None),
        # invalid second
        (1, 2, 3, 4, 5, None, 7, None, None),
        (1, 2, 3, 4, 5, -1, 7, None, None),
        (1, 2, 3, 4, 5, 60, 7, None, None),
        # invalid microsecond
        (1, 2, 3, 4, 5, 6, None, None, None),
        (1, 2, 3, 4, 5, 6, -1, None, None),
        (1, 2, 3, 4, 5, 6, 1_000_000, None, None),
        # with time zone
        (1, 2, 3, 4, 5, 6, 7, "UTC", datetime(1, 2, 3, 4, 5, 6, 7, tzinfo=UTC)),
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
        "with time zone",
    ],
)
def test_should_return_datetime(
    year: _ConvertibleToIntCell,
    month: _ConvertibleToIntCell,
    day: _ConvertibleToIntCell,
    hour: _ConvertibleToIntCell,
    minute: _ConvertibleToIntCell,
    second: _ConvertibleToIntCell,
    microsecond: _ConvertibleToIntCell,
    time_zone: str | None,
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
            time_zone=time_zone,
        ),
        expected,
    )


def test_should_raise_if_time_zone_is_invalid():
    column = Column("a", [None])
    with pytest.raises(ValueError, match="Invalid time zone"):
        column.transform(lambda _: Cell.datetime(1, 2, 3, time_zone="invalid"))
