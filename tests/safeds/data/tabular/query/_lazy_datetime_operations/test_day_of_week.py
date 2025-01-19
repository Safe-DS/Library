from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 1, 1, tzinfo=UTC), 1),
        (datetime(1, 1, 2, tzinfo=UTC), 2),
        (datetime(1, 1, 3, tzinfo=UTC), 3),
        (datetime(1, 1, 4, tzinfo=UTC), 4),
        (datetime(1, 1, 5, tzinfo=UTC), 5),
        (datetime(1, 1, 6, tzinfo=UTC), 6),
        (datetime(1, 1, 7, tzinfo=UTC), 7),
        (date(1, 1, 1), 1),
        (date(1, 1, 2), 2),
        (date(1, 1, 3), 3),
        (date(1, 1, 4), 4),
        (date(1, 1, 5), 5),
        (date(1, 1, 6), 6),
        (date(1, 1, 7), 7),
        (None, None),
    ],
    ids=[
        "datetime - Monday",
        "datetime - Tuesday",
        "datetime - Wednesday",
        "datetime - Thursday",
        "datetime - Friday",
        "datetime - Saturday",
        "datetime - Sunday",
        "date - Monday",
        "date - Tuesday",
        "date - Wednesday",
        "date - Thursday",
        "date - Friday",
        "date - Saturday",
        "date - Sunday",
        "None",
    ],
)
def test_should_extract_day_of_week(
    value: datetime | date | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.day_of_week(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
