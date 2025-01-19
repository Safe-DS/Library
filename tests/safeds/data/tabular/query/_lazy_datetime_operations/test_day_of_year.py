from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 1, 1, tzinfo=UTC), 1),
        (datetime(1, 12, 31, tzinfo=UTC), 365),
        (datetime(4, 12, 31, tzinfo=UTC), 366),
        (date(1, 1, 1), 1),
        (date(1, 12, 31), 365),
        (date(4, 12, 31), 366),
        (None, None),
    ],
    ids=[
        "datetime - first",
        "datetime - last in non-leap year",
        "datetime - last in leap year",
        "date - first",
        "date - last in non-leap year",
        "date - last in leap year",
        "None",
    ],
)
def test_should_extract_day_of_year(
    value: datetime | date | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.day_of_year(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
