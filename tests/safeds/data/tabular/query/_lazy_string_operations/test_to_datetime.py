from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from safeds.data.tabular.containers import Column
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works

DATETIME = datetime(1, 2, 3, 4, 5, 6)  # noqa: DTZ001
DATETIME_UTC = datetime(1, 2, 3, 4, 5, 6, tzinfo=ZoneInfo("UTC"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0001-02-03T04:05:06Z", DATETIME_UTC),
        (None, None),
    ],
    ids=[
        "datetime",
        "None",
    ],
)
def test_should_handle_iso_8601(value: str | None, expected: str | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_datetime(format="iso"),
        expected,
        type_if_none=ColumnType.string(),
    )


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        ("0001-02-03 04:05:06", "{Y}-{M}-{D} {h}:{m}:{s}", DATETIME),
        ("   1- 2- 3  4: 5: 6", "{_Y}-{_M}-{_D} {_h}:{_m}:{_s}", DATETIME),
        ("1-2-3 4:5:6", "{^Y}-{^M}-{^D} {^h}:{^m}:{^s}", DATETIME),
        ("invalid", "{Y}-{M}-{D} {h}:{m}:{s}", None),
    ],
    ids=[
        "{Y}-{M}-{D} {h}:{m}:{s}",
        "{_Y}-{_M}-{_D} {_h}:{_m}:{_s}",
        "{^Y}-{^M}-{^D} {^h}:{^m}:{^s}",
        "no match",
    ],
)
def test_should_handle_custom_format_string(value: str, format_: str, expected: datetime) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_datetime(format=format_),
        expected,
    )


@pytest.mark.parametrize(
    ("value", "format_", "expected"),
    [
        ("0001-02-03 04:05:06\\", "{Y}-{M}-{D} {h}:{m}:{s}\\", DATETIME),
        ("0001-02-03 04:05:06\\", "{Y}-{M}-{D} {h}:{m}:{s}\\\\", DATETIME),
        ("0001-02-03 04:05:06{", "{Y}-{M}-{D} {h}:{m}:{s}\\{", DATETIME),
        ("0001-02-03 04:05:06%", "{Y}-{M}-{D} {h}:{m}:{s}%", DATETIME),
        ("0001-02-03 04:05:06\n", "{Y}-{M}-{D} {h}:{m}:{s}\n", DATETIME),
        ("0001-02-03 04:05:06\t", "{Y}-{M}-{D} {h}:{m}:{s}\t", DATETIME),
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
def test_should_handle_escape_sequences(value: str, format_: str, expected: datetime) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.str.to_datetime(format=format_),
        expected,
    )


def test_should_raise_for_unclosed_specifier() -> None:
    column = Column("a", ["0001-02-03 04:05:06"])
    with pytest.raises(ValueError, match="Unclosed specifier"):
        column.transform(lambda cell: cell.str.to_datetime(format="{Y"))


@pytest.mark.parametrize(
    "format_",
    [
        "{invalid}",
    ],
    ids=[
        "globally invalid",
    ],
)
def test_should_raise_for_invalid_specifier(format_: str) -> None:
    column = Column("a", ["0001-02-03"])
    with pytest.raises(ValueError, match="Invalid specifier"):
        column.transform(lambda cell: cell.str.to_datetime(format=format_))
