import datetime

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("expected", "input_date", "format_string"),
    [
        ("2022/01/09 23:29:01", datetime.datetime(2022, 1, 9, 23, 29, 1, tzinfo=datetime.UTC), "%Y/%m/%d %H:%M:%S"),
        ("2022:01:09 23/29/01", datetime.datetime(2022, 1, 9, 23, 29, 1, tzinfo=datetime.UTC), "%Y:%m:%d %H/%M/%S"),
    ],
    ids=[
        "ISO datetime",
        "ISO datetime format",
    ],
)
def test_should_parse_date_to_string(input_date: datetime.date, expected: bool, format_string: str) -> None:
    assert_cell_operation_works(input_date, lambda cell: cell.dt.datetime_to_string(format_string), expected)


@pytest.mark.parametrize(
    ("expected", "input_date"),
    [
        ("Invalid format string", datetime.datetime(2022, 1, 9, 23, 29, 1, tzinfo=datetime.UTC)),
    ],
    ids=[
        "ISO datetime",
    ],
)
def test_should_raise_value_error_when_input_date_is_invalid(input_date: datetime.date, expected: bool) -> None:
    with pytest.raises(ValueError, match=expected):
        assert_cell_operation_works(input_date, lambda cell: cell.dt.datetime_to_string("%9"), expected)
