from datetime import timedelta
from typing import Literal

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        # ISO 8601 format
        (timedelta(weeks=1), "iso", "P7D"),
        (timedelta(weeks=-1), "iso", "-P7D"),
        (timedelta(days=1), "iso", "P1D"),
        (timedelta(days=-1), "iso", "-P1D"),
        (timedelta(hours=1), "iso", "PT1H"),
        (timedelta(hours=-1), "iso", "-PT1H"),
        (timedelta(minutes=1), "iso", "PT1M"),
        (timedelta(minutes=-1), "iso", "-PT1M"),
        (timedelta(seconds=1), "iso", "PT1S"),
        (timedelta(seconds=-1), "iso", "-PT1S"),
        (timedelta(milliseconds=1), "iso", "PT0.001S"),
        (timedelta(milliseconds=-1), "iso", "-PT0.001S"),
        (timedelta(microseconds=1), "iso", "PT0.000001S"),
        (timedelta(microseconds=-1), "iso", "-PT0.000001S"),
        (
            timedelta(weeks=1, days=1, hours=1, minutes=1, seconds=1, milliseconds=1, microseconds=1),
            "iso",
            "P8DT1H1M1.001001S",
        ),
        (
            timedelta(weeks=-1, days=-1, hours=-1, minutes=-1, seconds=-1, milliseconds=-1, microseconds=-1),
            "iso",
            "-P8DT1H1M1.001001S",
        ),
        (
            timedelta(weeks=1, days=-1, hours=1, minutes=-1, seconds=1, milliseconds=-1, microseconds=1),
            "iso",
            "P6DT59M0.999001S",
        ),
        (None, "iso", None),
        # Pretty format
        (timedelta(weeks=1), "pretty", "7d"),
        (timedelta(weeks=-1), "pretty", "-7d"),
        (timedelta(days=1), "pretty", "1d"),
        (timedelta(days=-1), "pretty", "-1d"),
        (timedelta(hours=1), "pretty", "1h"),
        (timedelta(hours=-1), "pretty", "-1h"),
        (timedelta(minutes=1), "pretty", "1m"),
        (timedelta(minutes=-1), "pretty", "-1m"),
        (timedelta(seconds=1), "pretty", "1s"),
        (timedelta(seconds=-1), "pretty", "-1s"),
        (timedelta(milliseconds=1), "pretty", "1ms"),
        (timedelta(milliseconds=-1), "pretty", "-1ms"),
        (timedelta(microseconds=1), "pretty", "1µs"),
        (timedelta(microseconds=-1), "pretty", "-1µs"),
        (
            timedelta(weeks=1, days=1, hours=1, minutes=1, seconds=1, milliseconds=1, microseconds=1),
            "pretty",
            "8d 1h 1m 1s 1001µs",
        ),
        (
            timedelta(weeks=-1, days=-1, hours=-1, minutes=-1, seconds=-1, milliseconds=-1, microseconds=-1),
            "pretty",
            "-8d -1h -1m -1s -1001µs",
        ),
        (
            timedelta(weeks=1, days=-1, hours=1, minutes=-1, seconds=1, milliseconds=-1, microseconds=1),
            "pretty",
            "6d 59m 999001µs",
        ),
        (None, "pretty", None),
    ],
    ids=[
        # ISO 8601 format
        "iso - positive weeks",
        "iso - negative weeks",
        "iso - positive days",
        "iso - negative days",
        "iso - positive hours",
        "iso - negative hours",
        "iso - positive minutes",
        "iso - negative minutes",
        "iso - positive seconds",
        "iso - negative seconds",
        "iso - positive milliseconds",
        "iso - negative milliseconds",
        "iso - positive microseconds",
        "iso - negative microseconds",
        "iso - all positive",
        "iso - all negative",
        "iso - mixed",
        "iso - None",
        # Pretty format
        "pretty - positive weeks",
        "pretty - negative weeks",
        "pretty - positive days",
        "pretty - negative days",
        "pretty - positive hours",
        "pretty - negative hours",
        "pretty - positive minutes",
        "pretty - negative minutes",
        "pretty - positive seconds",
        "pretty - negative seconds",
        "pretty - positive milliseconds",
        "pretty - negative milliseconds",
        "pretty - positive microseconds",
        "pretty - negative microseconds",
        "pretty - all positive",
        "pretty - all negative",
        "pretty - mixed",
        "pretty - None",
    ],
)
def test_should_return_string_representation(
    value: timedelta | None,
    format_: Literal["iso", "pretty"],
    expected: str | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dur.to_string(format=format_),
        expected,
        type_if_none=ColumnType.duration(),
    )
