import datetime

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("expected", "input_date"),
    [
        ("2022/01/09 23:29:01", datetime.datetime(2022, 1, 9, 23, 29, 1, tzinfo=datetime.UTC)),
    ],
    ids=[
        "ISO datetime",
    ],
)
def test_should_parse_date_to_string(input_date: datetime.date, expected: bool) -> None:
    assert_cell_operation_works(input_date, lambda cell: cell.dt.datetime_to_string(), expected)
