from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 2, 3, tzinfo=UTC), 3),
        (date(1, 2, 3), 3),
        (None, None),
    ],
    ids=[
        "datetime",
        "date",
        "None",
    ],
)
def test_should_extract_day(
    value: datetime | date | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.day(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
