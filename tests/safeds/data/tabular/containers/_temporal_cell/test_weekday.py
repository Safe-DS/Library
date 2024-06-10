import datetime

import pytest

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("expected", "input_date"),
    [
        (4, datetime.datetime(2023, 3, 9, 23, 29, 1,tzinfo=datetime.timezone.utc)),
        (6, datetime.date(2022, 1, 1)),
    ],
    ids=[
        "ISO datetime",
        "ISO date",
    ],
)
def test_get_weekday(input_date: datetime.date, expected: bool) -> None:
    assert_cell_operation_works(input_date, lambda cell: cell.dt.weekday(), expected)

