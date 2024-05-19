import datetime

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("", None),
        ("2022-01-09T23:29:01Z", datetime.datetime(2022, 1, 9, 23, 29, 1, tzinfo=datetime.UTC)),
        ("abc", None),
    ],
    ids=[
        "empty",
        "ISO datetime",
        "invalid string",
    ],
)
def test_should_parse_datetimes(string: str, expected: bool) -> None:
    assert_cell_operation_works(string, lambda cell: cell.string.to_datetime(), expected)
