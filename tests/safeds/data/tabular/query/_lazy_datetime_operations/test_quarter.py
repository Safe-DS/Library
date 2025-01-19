from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 1, 1, tzinfo=UTC), 1),
        (datetime(1, 3, 31, tzinfo=UTC), 1),
        (datetime(1, 4, 1, tzinfo=UTC), 2),
        (datetime(1, 6, 30, tzinfo=UTC), 2),
        (datetime(1, 7, 1, tzinfo=UTC), 3),
        (datetime(1, 9, 30, tzinfo=UTC), 3),
        (datetime(1, 10, 1, tzinfo=UTC), 4),
        (datetime(1, 12, 31, tzinfo=UTC), 4),
        (date(1, 1, 1), 1),
        (date(1, 3, 31), 1),
        (date(1, 4, 1), 2),
        (date(1, 6, 30), 2),
        (date(1, 7, 1), 3),
        (date(1, 9, 30), 3),
        (date(1, 10, 1), 4),
        (date(1, 12, 31), 4),
        (None, None),
    ],
    ids=[
        "datetime - first day of first quarter",
        "datetime - last day of first quarter",
        "datetime - first day of second quarter",
        "datetime - last day of second quarter",
        "datetime - first day of third quarter",
        "datetime - last day of third quarter",
        "datetime - first day of fourth quarter",
        "datetime - last day of fourth quarter",
        "date - first day of first quarter",
        "date - last day of first quarter",
        "date - first day of second quarter",
        "date - last day of second quarter",
        "date - first day of third quarter",
        "date - last day of third quarter",
        "date - first day of fourth quarter",
        "date - last day of fourth quarter",
        "None",
    ],
)
def test_should_extract_quarter(
    value: datetime | date | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.quarter(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
