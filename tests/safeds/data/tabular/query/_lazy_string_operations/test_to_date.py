from datetime import date

import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works

DATE = date(1, 2, 3)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0001-02-03", DATE),
        (None, None),
    ],
    ids=[
        "date",
        "None",
    ],
)
def test_should_handle_iso_8601(value: str | None, expected: str | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_date(format="iso"),
        expected,
        type_if_none=ColumnType.string(),
    )


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        ("0001-02-03", "{Y}-{M}-{D}", DATE),
        ("   1- 2- 3", "{_Y}-{_M}-{_D}", DATE),
        ("1-2-3", "{^Y}-{^M}-{^D}", DATE),
        ("01", "{Y99}", date(2001, 1, 1)),  # weird polars behavior
        (" 1", "{_Y99}", None),
        ("1", "{^Y99}", None),
        ("0001-February-03", "{Y}-{M-full}-{D}", DATE),
        ("0001-Feb-03", "{Y}-{M-short}-{D}", DATE),
        ("0001-02-03 05| 5|5", "{Y}-{M}-{D} {W}|{_W}|{^W}", DATE),
        ("0001-02-03 6|Saturday|Sat", "{Y}-{M}-{D} {DOW}|{DOW-full}|{DOW-short}", DATE),
        ("0001/034", "{Y}/{DOY}", DATE),
        ("   1/ 34", "{Y}/{_DOY}", DATE),
        ("   1/034", "{Y}/{^DOY}", DATE),
        ("0001-02-03 0001", "{Y}-{M}-{D} {Y}", DATE),
        ("0001-02-03 0004", "{Y}-{M}-{D} {Y}", date(4, 2, 3)),  # weird polars behavior
        ("0001-02-03 01", "{Y}-{M}-{D} {Y99}", date(2001, 2, 3)),  # weird polars behavior
        ("0001-02-03 04", "{Y}-{M}-{D} {Y99}", date(2004, 2, 3)),  # weird polars behavior
        ("24:00:00", "{Y}-{M}-{D}", None),
        ("invalid", "{Y}-{M}-{D}", None),
    ],
    ids=[
        "{Y}-{M}-{D}",
        "{_Y}-{_M}-{_D}",
        "{^Y}-{^M}-{^D}",
        "{Y99}",
        "{_Y99}",
        "{^Y99}",
        "{Y}-{M-full}-{D}",
        "{Y}-{M-short}-{D}",
        "week number",
        "day of the week",
        "{Y}/{DOY}",
        "{_Y}/{_DOY}",
        "{^Y}/{^DOY}",
        "duplicate field, same value",
        "duplicate field, different value",
        "similar field, same value",
        "similar field, different value",
        "out of bounds",
        "no match",
    ],
)
def test_should_handle_custom_format_string(value: str, format_: str, expected: date) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_date(format=format_),
        expected,
    )


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        ("0001-02-03\\", "{Y}-{M}-{D}\\", DATE),
        ("0001-02-03\\", "{Y}-{M}-{D}\\\\", DATE),
        ("0001-02-03{", "{Y}-{M}-{D}\\{", DATE),
        ("0001-02-03%", "{Y}-{M}-{D}%", DATE),
        ("0001-02-03\n", "{Y}-{M}-{D}\n", DATE),
        ("0001-02-03\t", "{Y}-{M}-{D}\t", DATE),
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
def test_should_handle_escape_sequences(value: str, format_: str, expected: date) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_date(format=format_),
        expected,
    )


def test_should_raise_for_unclosed_specifier() -> None:
    column = Column("a", ["0001-02-03"])
    with pytest.raises(ValueError, match="Unclosed specifier"):
        column.transform(lambda cell: cell.str.to_date(format="{Y"))


@pytest.mark.parametrize(
    "format_",
    [
        "{invalid}",
        "{m}",
    ],
    ids=[
        "globally invalid",
        "invalid for date",
    ],
)
def test_should_raise_for_invalid_specifier(format_: str) -> None:
    column = Column("a", ["0001-02-03"])
    with pytest.raises(ValueError, match="Invalid specifier"):
        column.transform(lambda cell: cell.str.to_date(format=format_))
