from datetime import UTC, date, datetime, time

import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from safeds.exceptions import LazyComputationError
from tests.helpers import assert_cell_operation_works

DATETIME = datetime(1, 2, 3, 4, 5, 6, 7, tzinfo=UTC)
DATE = date(1, 2, 3)
TIME = time(4, 5, 6, 7)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 2, 3, 4, 5, 6, 7), "0001-02-03T04:05:06.000007"),  # noqa: DTZ001
        (DATETIME, "0001-02-03T04:05:06.000007+00:00"),
        (DATE, "0001-02-03"),
        (TIME, "04:05:06.000007"),
        (None, None),
    ],
    ids=[
        "datetime without time zone",
        "datetime with time zone",
        "date",
        "time",
        "None",
    ],
)
def test_should_handle_iso_8601(value: datetime | date | time | None, expected: str | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.to_string(format="iso"),
        expected,
        type_if_none=ColumnType.datetime(),
    )


@pytest.mark.parametrize(
    "value",
    [
        DATETIME,
        DATE,
    ],
    ids=[
        "datetime",
        "date",
    ],
)
class TestDateSpecifiers:
    @pytest.mark.parametrize(
        ("format_", "expected"),
        [
            ("{Y}", "0001"),
            ("{_Y}", "   1"),
            ("{^Y}", "1"),
            ("{Y99}", "01"),
            ("{_Y99}", " 1"),
            ("{^Y99}", "1"),
            ("{M}", "02"),
            ("{_M}", " 2"),
            ("{^M}", "2"),
            ("{M-full}", "February"),
            ("{M-short}", "Feb"),
            ("{W}", "05"),
            ("{_W}", " 5"),
            ("{^W}", "5"),
            ("{D}", "03"),
            ("{_D}", " 3"),
            ("{^D}", "3"),
            ("{DOW}", "6"),
            ("{DOW-full}", "Saturday"),
            ("{DOW-short}", "Sat"),
            ("{DOY}", "034"),
            ("{_DOY}", " 34"),
            ("{^DOY}", "34"),
        ],
        ids=[
            "{Y}",
            "{_Y}",
            "{^Y}",
            "{Y99}",
            "{_Y99}",
            "{^Y99}",
            "{M}",
            "{_M}",
            "{^M}",
            "{M-full}",
            "{M-short}",
            "{W}",
            "{_W}",
            "{^W}",
            "{D}",
            "{_D}",
            "{^D}",
            "{DOW}",
            "{DOW-full}",
            "{DOW-short}",
            "{DOY}",
            "{_DOY}",
            "{^DOY}",
        ],
    )
    def test_should_be_replaced_with_correct_string(self, value: datetime | date, format_: str, expected: str) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.dt.to_string(format=format_),
            expected,
        )


@pytest.mark.parametrize(
    "value",
    [
        DATETIME,
        TIME,
    ],
    ids=[
        "datetime",
        "time",
    ],
)
class TestTimeSpecifiers:
    @pytest.mark.parametrize(
        ("format_", "expected"),
        [
            ("{h}", "04"),
            ("{_h}", " 4"),
            ("{^h}", "4"),
            ("{h12}", "04"),
            ("{_h12}", " 4"),
            ("{^h12}", "4"),
            ("{m}", "05"),
            ("{_m}", " 5"),
            ("{^m}", "5"),
            ("{s}", "06"),
            ("{_s}", " 6"),
            ("{^s}", "6"),
            ("{.f}", ".000007"),
            ("{ms}", "000"),
            ("{us}", "000007"),
            ("{ns}", "000007000"),
            ("{AM/PM}", "AM"),
            ("{am/pm}", "am"),
        ],
        ids=[
            "{h}",
            "{_h}",
            "{^h}",
            "{h12}",
            "{_h12}",
            "{^h12}",
            "{m}",
            "{_m}",
            "{^m}",
            "{s}",
            "{_s}",
            "{^s}",
            "{.f}",
            "{ms}",
            "{us}",
            "{ns}",
            "{AM/PM}",
            "{am/pm}",
        ],
    )
    def test_should_be_replaced_with_correct_string(self, value: datetime | time, format_: str, expected: str) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.dt.to_string(format=format_),
            expected,
        )


@pytest.mark.parametrize(
    "value",
    [
        DATETIME,
    ],
    ids=[
        "datetime",
    ],
)
class TestDateTimeSpecifiers:
    @pytest.mark.parametrize(
        ("format_", "expected"),
        [
            ("{z}", "+0000"),
            ("{:z}", "+00:00"),
            ("{u}", "-62132730894"),
        ],
        ids=[
            "{z}",
            "{:z}",
            "{u}",
        ],
    )
    def test_should_be_replaced_with_correct_string(self, value: datetime, format_: str, expected: str) -> None:
        assert_cell_operation_works(
            value,
            lambda cell: cell.dt.to_string(format=format_),
            expected,
        )


@pytest.mark.parametrize(
    ("format_", "expected"),
    [
        ("\\", "\\"),
        ("\\\\", "\\"),
        ("\\{", "{"),
        ("%", "%"),
        ("\n", "\n"),
        ("\t", "\t"),
    ],
    ids=[
        "backslash at end",
        "escaped backslash",
        "escaped open curly brace",
        "percent",
        "newline",
        "tab",
    ],
)
def test_should_handle_escape_sequences(format_: str, expected: date | time | None) -> None:
    assert_cell_operation_works(
        DATETIME,
        lambda cell: cell.dt.to_string(format=format_),
        expected,
    )


def test_should_raise_for_unclosed_specifier() -> None:
    column = Column("a", [DATETIME])
    with pytest.raises(ValueError, match="Unclosed specifier"):
        column.transform(lambda cell: cell.dt.to_string(format="{Y"))


def test_should_raise_for_globally_invalid_specifier() -> None:
    column = Column("a", [DATETIME])
    with pytest.raises(ValueError, match="Invalid specifier"):
        column.transform(lambda cell: cell.dt.to_string(format="{invalid}"))


@pytest.mark.parametrize(
    ("value", "format_"),
    [
        (DATE, "{h}"),
        pytest.param(
            TIME,
            "{Y}",
            marks=pytest.mark.skip("polars panics in this case (https://github.com/pola-rs/polars/issues/19853)."),
        ),
    ],
    ids=[
        "invalid for date",
        "invalid for time",
    ],
)
def test_should_raise_for_specifier_that_is_invalid_for_type(value: date | time | None, format_: str) -> None:
    # TODO: This is not the ideal behavior. Once https://github.com/Safe-DS/Library/issues/860 is resolved, we should
    #  do our own validation to raise an error that knows our own specifiers.
    column = Column("a", [value])
    lazy_result = column.transform(lambda cell: cell.dt.to_string(format=format_))
    with pytest.raises(LazyComputationError):
        lazy_result.get_value(0)
