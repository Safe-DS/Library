from datetime import time

import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works

NO_FRACTIONAL = time(4, 5, 6)
WITH_MILLISECOND = time(4, 5, 6, 7000)
WITH_MICROSECOND = time(4, 5, 6, 7)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("04:05:06", NO_FRACTIONAL),
        ("04:05:06.007", WITH_MILLISECOND),
        ("04:05:06.000007", WITH_MICROSECOND),
        (None, None),
    ],
    ids=[
        "time without fractional seconds",
        "time with milliseconds",
        "time with microseconds",
        "None",
    ],
)
def test_should_handle_iso_8601(value: str | None, expected: str | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_time(format="iso"),
        expected,
        type_if_none=ColumnType.string(),
    )


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        ("04:05:06", "{h}:{m}:{s}", NO_FRACTIONAL),
        (" 4: 5: 6", "{_h}:{_m}:{_s}", NO_FRACTIONAL),
        ("4:5:6", "{^h}:{^m}:{^s}", NO_FRACTIONAL),
        ("04:05:06 am", "{h12}:{m}:{s} {am/pm}", NO_FRACTIONAL),
        (" 4: 5: 6 AM", "{_h12}:{m}:{s} {AM/PM}", NO_FRACTIONAL),
        ("4:5:6 AM", "{^h12}:{m}:{s} {AM/PM}", NO_FRACTIONAL),
        ("04:05:06 .000007", "{h}:{m}:{s} {.f}", WITH_MICROSECOND),
        ("04:05:06 007", "{h}:{m}:{s} {ms}", WITH_MILLISECOND),
        ("04:05:06 000007", "{h}:{m}:{s} {us}", WITH_MICROSECOND),
        ("04:05:06 000007000", "{h}:{m}:{s} {ns}", WITH_MICROSECOND),
        ("04", "{h}", None),
        ("05", "{m}", None),
        ("04:05:06 04", "{h}:{m}:{s} {h}", NO_FRACTIONAL),
        ("04:05:06 07", "{h}:{m}:{s} {h}", None),
        ("04:05:06 04", "{h}:{m}:{s} {h12}", NO_FRACTIONAL),
        ("04:05:06 07", "{h}:{m}:{s} {h12}", None),
        ("24:00:00", "{h}:{m}:{s}", None),
        ("invalid", "{h}:{m}:{s}", None),
    ],
    ids=[
        "{h}:{m}:{s}",
        "{_h}:{_m}:{_s}",
        "{^h}:{^m}:{^s}",
        "{h12}:{m}:{s} {am/pm}",
        "{_h12}:{m}:{s} {am/pm}",
        "{^h12}:{m}:{s} {AM/PM}",
        "{.f}",
        "{ms}",
        "{us}",
        "{ns}",
        "no minute",
        "no hour",
        "duplicate field, same value",
        "duplicate field, different value",
        "similar field, same value",
        "similar field, different value",
        "out of bounds",
        "no match",
    ],
)
def test_should_handle_custom_format_string(value: str, format_: str, expected: time) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_time(format=format_),
        expected,
    )


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        ("04:05\\", "{h}:{m}\\", time(4, 5)),
        ("04:05\\", "{h}:{m}\\\\", time(4, 5)),
        ("04:05{", "{h}:{m}\\{", time(4, 5)),
        ("04:05%", "{h}:{m}%", time(4, 5)),
        ("04:05\n", "{h}:{m}\n", time(4, 5)),
        ("04:05\t", "{h}:{m}\t", time(4, 5)),
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
def test_should_handle_escape_sequences(value: str, format_: str, expected: time) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_time(format=format_),
        expected,
    )


def test_should_raise_for_unclosed_specifier() -> None:
    column = Column("a", ["04:05:06"])
    with pytest.raises(ValueError, match="Unclosed specifier"):
        column.transform(lambda cell: cell.str.to_time(format="{m"))


@pytest.mark.parametrize(
    "format_",
    [
        "{invalid}",
        "{Y}",
    ],
    ids=[
        "globally invalid",
        "invalid for time",
    ],
)
def test_should_raise_for_invalid_specifier(format_: str) -> None:
    column = Column("a", ["04:05:06"])
    with pytest.raises(ValueError, match="Invalid specifier"):
        column.transform(lambda cell: cell.str.to_time(format=format_))
