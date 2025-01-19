from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1999, 1, 1, tzinfo=UTC), False),
        (datetime(1996, 1, 1, tzinfo=UTC), True),
        (datetime(1900, 1, 1, tzinfo=UTC), False),
        (datetime(2000, 1, 1, tzinfo=UTC), True),
        (date(1999, 1, 1), False),
        (date(1996, 1, 1), True),
        (date(1900, 1, 1), False),
        (date(2000, 1, 1), True),
        (None, None),
    ],
    ids=[
        "datetime - not divisible by 4",
        "datetime - divisible by 4",
        "datetime - divisible by 100",
        "datetime - divisible by 400",
        "date - not divisible by 4",
        "date - divisible by 4",
        "date - divisible by 100",
        "date - divisible by 400",
        "None",
    ],
)
def test_should_return_unix_timestamp(
    value: datetime | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.is_in_leap_year(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
