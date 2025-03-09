from datetime import UTC, datetime

import pytest

from safeds._typing import _ConvertibleToIntCell
from safeds.data.tabular.containers import Cell, Column
from safeds.exceptions import LazyComputationError
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
        # with None components
        (None, 2, 3, 4, 5, 6, 7, None, None),
        (1, None, 3, 4, 5, 6, 7, None, None),
        (1, 2, None, 4, 5, 6, 7, None, None),
        (1, 2, 3, None, 5, 6, 7, None, None),
        (1, 2, 3, 4, None, 6, 7, None, None),
        (1, 2, 3, 4, 5, None, 7, None, None),
        (1, 2, 3, 4, 5, 6, None, None, None),
        # with time zone
        (1, 2, 3, 4, 5, 6, 7, "UTC", datetime(1, 2, 3, 4, 5, 6, 7, tzinfo=UTC)),
    ],
    ids=[
        "int components",
        "cell components",
        "year is None",
        "month is None",
        "day is None",
        "hour is None",
        "minute is None",
        "second is None",
        "microsecond is None",
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


@pytest.mark.parametrize(
    ("year", "month", "day", "hour", "minute", "second", "microsecond"),
    [
        # invalid month
        (1, 0, 3, 4, 5, 6, 7),
        (1, 13, 3, 4, 5, 6, 7),
        # invalid day
        (1, 2, 0, 4, 5, 6, 7),
        (1, 2, 32, 4, 5, 6, 7),
        # invalid hour
        (1, 2, 3, -1, 5, 6, 7),
        (1, 2, 3, 24, 5, 6, 7),
        # invalid minute
        (1, 2, 3, 4, -1, 6, 7),
        (1, 2, 3, 4, 60, 6, 7),
        # invalid second
        (1, 2, 3, 4, 5, -1, 7),
        (1, 2, 3, 4, 5, 60, 7),
        # invalid microsecond
        (1, 2, 3, 4, 5, 6, -1),
        pytest.param(
            1,
            2,
            3,
            4,
            5,
            6,
            1_000_000,
            marks=pytest.mark.xfail(reason="https://github.com/pola-rs/polars/issues/21664"),
        ),
    ],
    ids=[
        "month is too low",
        "month is too high",
        "day is too low",
        "day is too high",
        "hour is too low",
        "hour is too high",
        "minute is too low",
        "minute is too high",
        "second is too low",
        "second is too high",
        "microsecond is too low",
        "microsecond is too high",
    ],
)
def test_should_raise_for_invalid_components(
    year: _ConvertibleToIntCell,
    month: _ConvertibleToIntCell,
    day: _ConvertibleToIntCell,
    hour: _ConvertibleToIntCell,
    minute: _ConvertibleToIntCell,
    second: _ConvertibleToIntCell,
    microsecond: _ConvertibleToIntCell,
) -> None:
    column = Column("col1", [None])
    with pytest.raises(LazyComputationError):
        column.transform(
            lambda _: Cell.datetime(year, month, day, hour, minute, second, microsecond=microsecond),
        ).get_value(0)


def test_should_raise_if_time_zone_is_invalid() -> None:
    column = Column("col1", [None])
    with pytest.raises(ValueError, match="Invalid time zone"):
        column.transform(lambda _: Cell.datetime(1, 2, 3, 0, 0, 0, time_zone="invalid"))
