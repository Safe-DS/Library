import datetime

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("expected", "input_date"),
    [
        (9, datetime.datetime(2022, 1, 9, 23, 29, 1,tzinfo=datetime.timezone.utc)),
        (1, datetime.date(2022, 1, 1)),
    ],
    ids=[
        "ISO datetime",
        "ISO date",
    ],
)
def test_get_day(input_date: datetime.date, expected: bool) -> None:
    assert_cell_operation_works(input_date, lambda cell: cell.dt.day(), expected)

