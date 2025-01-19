from datetime import UTC, date, datetime, time

import pytest
from helpers import assert_cell_operation_works
from polars.polars import ComputeError, PanicException

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType

DATETIME = datetime(1, 2, 3, 4, 5, 6, 7, tzinfo=UTC)
DATE = date(1, 2, 3)
TIME = time(4, 5, 6, 7)


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        (datetime(1, 2, 3, 4, 5, 6, 7), "iso", "0001-02-03T04:05:06.000007"),  # noqa: DTZ001
        (DATETIME, "iso", "0001-02-03T04:05:06.000007+00:00"),
        (None, "iso", None),
    ],
    ids=[
        "iso, no time zone",
        "iso, with time zone",
        "None",
    ],
)
def test_should_stringify_datetime(value: datetime | None, format_: str, expected: str | None):
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.to_string(format=format_),
        expected,
        type_if_none=ColumnType.datetime(),
    )


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        (DATE, "iso", "0001-02-03"),
        (None, "iso", None),
    ],
    ids=[
        "iso",
        "None",
    ],
)
def test_should_stringify_date(value: date | None, format_: str, expected: str | None):
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.to_string(format=format_),
        expected,
        type_if_none=ColumnType.date(),
    )


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        (TIME, "iso", "04:05:06.000007"),
        (None, "iso", None),
    ],
    ids=[
        "iso",
        "None",
    ],
)
def test_should_stringify_time(value: time | None, format_: str, expected: str | None):
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.to_string(format=format_),
        expected,
        type_if_none=ColumnType.time(),
    )


def test_should_raise_for_unclosed_specifier():
    column = Column("a", [DATETIME])
    with pytest.raises(ValueError, match="Unclosed specifier"):
        column.transform(lambda cell: cell.dt.to_string(format="{Y"))


def test_should_raise_for_globally_invalid_specifier():
    column = Column("a", [DATETIME])
    with pytest.raises(ValueError, match="Invalid specifier"):
        column.transform(lambda cell: cell.dt.to_string(format="{invalid}"))


@pytest.mark.parametrize(
    ("value", "format_"),
    [
        (DATE, "{h}"),
        (TIME, "{Y}"),
    ],
    ids=[
        "invalid for date",
        "invalid for time",
    ],
)
def test_should_raise_for_specifier_that_is_invalid_for_type(value: date | time | None, format_: str):
    # TODO: This is not the ideal behavior. Once https://github.com/Safe-DS/Library/issues/860 is resolved, we should
    #  do our own validation to raise an error that knows our own specifiers.
    column = Column("a", [value])
    with pytest.raises((ComputeError, PanicException)):
        column.transform(lambda cell: cell.dt.to_string(format=format_))
