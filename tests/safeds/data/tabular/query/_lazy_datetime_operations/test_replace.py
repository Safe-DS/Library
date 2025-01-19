from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works

DATETIME = datetime(1, 2, 3, 4, 5, 6, 7, tzinfo=UTC)
DATE = date(1, 2, 3)


@pytest.mark.parametrize(
    ("value", "year", "month", "day", "hour", "minute", "second", "microsecond", "expected"),
    [
        # datetime - change year
        (DATETIME, 10, None, None, None, None, None, None, datetime(10, 2, 3, 4, 5, 6, 7, tzinfo=UTC)),
        # datetime - change month (valid)
        (DATETIME, None, 10, None, None, None, None, None, datetime(1, 10, 3, 4, 5, 6, 7, tzinfo=UTC)),
        # datetime - change month (invalid)
        (DATETIME, None, 13, None, None, None, None, None, None),
        # datetime - change day (valid)
        (DATETIME, None, None, 10, None, None, None, None, datetime(1, 2, 10, 4, 5, 6, 7, tzinfo=UTC)),
        # datetime - change day (invalid)
        (DATETIME, None, None, 32, None, None, None, None, None),
        # datetime - change hour (valid)
        (DATETIME, None, None, None, 10, None, None, None, datetime(1, 2, 3, 10, 5, 6, 7, tzinfo=UTC)),
        # datetime - change hour (invalid)
        (DATETIME, None, None, None, 24, None, None, None, None),
        # datetime - change minute (valid)
        (DATETIME, None, None, None, None, 10, None, None, datetime(1, 2, 3, 4, 10, 6, 7, tzinfo=UTC)),
        # datetime - change minute (invalid)
        (DATETIME, None, None, None, None, 60, None, None, None),
        # datetime - change second (valid)
        (DATETIME, None, None, None, None, None, 10, None, datetime(1, 2, 3, 4, 5, 10, 7, tzinfo=UTC)),
        # datetime - change second (invalid)
        (DATETIME, None, None, None, None, None, 60, None, None),
        # datetime - change microsecond (valid)
        (DATETIME, None, None, None, None, None, None, 10, datetime(1, 2, 3, 4, 5, 6, 10, tzinfo=UTC)),
        # datetime - change microsecond (invalid)
        (DATETIME, None, None, None, None, None, None, 1000000, None),
        # date - change year
        (DATE, 10, None, None, None, None, None, None, date(10, 2, 3)),
        # date - change month (valid)
        (DATE, None, 10, None, None, None, None, None, date(1, 10, 3)),
        # date - change month (invalid)
        (DATE, None, 13, None, None, None, None, None, None),
        # date - change day (valid)
        (DATE, None, None, 10, None, None, None, None, date(1, 2, 10)),
        # date - change day (invalid)
        (DATE, None, None, 32, None, None, None, None, None),
        # date - change hour (valid)
        (DATE, None, None, None, 10, None, None, None, DATE),
        # date - change hour (invalid)
        (DATE, None, None, None, 24, None, None, None, DATE),
        # date - change minute (valid)
        (DATE, None, None, None, None, 10, None, None, DATE),
        # date - change minute (invalid)
        (DATE, None, None, None, None, 60, None, None, DATE),
        # date - change second (valid)
        (DATE, None, None, None, None, None, 10, None, DATE),
        # date - change second (invalid)
        (DATE, None, None, None, None, None, 60, None, DATE),
        # date - change microsecond (valid)
        (DATE, None, None, None, None, None, None, 10, DATE),
        # date - change microsecond (invalid)
        (DATE, None, None, None, None, None, None, 1000000, DATE),
        # None
        (None, None, None, None, None, None, None, None, None),
    ],
    ids=[
        # datetime
        "datetime - change year",
        "datetime - change month (valid)",
        "datetime - change month (invalid)",
        "datetime - change day (valid)",
        "datetime - change day (invalid)",
        "datetime - change hour (valid)",
        "datetime - change hour (invalid)",
        "datetime - change minute (valid)",
        "datetime - change minute (invalid)",
        "datetime - change second (valid)",
        "datetime - change second (invalid)",
        "datetime - change microsecond (valid)",
        "datetime - change microsecond (invalid)",
        # date
        "date - change year",
        "date - change month (valid)",
        "date - change month (invalid)",
        "date - change day (valid)",
        "date - change day (invalid)",
        "date - change hour (valid)",
        "date - change hour (invalid)",
        "date - change minute (valid)",
        "date - change minute (invalid)",
        "date - change second (valid)",
        "date - change second (invalid)",
        "date - change microsecond (valid)",
        "date - change microsecond (invalid)",
        # None
        "None",
    ],
)
def test_should_replace_components(
    value: datetime | date | None,
    year: int | None,
    month: int | None,
    day: int | None,
    hour: int | None,
    minute: int | None,
    second: int | None,
    microsecond: int | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.replace(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond,
        ),
        expected,
        type_if_none=ColumnType.datetime(),
    )
