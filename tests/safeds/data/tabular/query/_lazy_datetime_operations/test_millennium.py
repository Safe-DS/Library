from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 1, 1, tzinfo=UTC), 1),
        (datetime(1000, 12, 31, tzinfo=UTC), 1),
        (datetime(1001, 1, 1, tzinfo=UTC), 2),
        (date(1, 1, 1), 1),
        (date(1000, 12, 31), 1),
        (date(1001, 1, 1), 2),
        (None, None),
    ],
    ids=[
        "datetime - first day of first millennium",
        "datetime - last day of first millennium",
        "datetime - first day of second millennium",
        "date - first day of first millennium",
        "date - last day of first millennium",
        "date - first day of second millennium",
        "None",
    ],
)
def test_should_extract_millennium(
    value: datetime | date | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.millennium(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
