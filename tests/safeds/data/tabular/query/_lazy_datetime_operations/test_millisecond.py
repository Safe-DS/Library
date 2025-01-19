from datetime import UTC, datetime, time

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 2, 3, 4, 5, 6, 7000, tzinfo=UTC), 7),
        (datetime(1, 2, 3, 4, 5, 6, 999, tzinfo=UTC), 0),
        (time(4, 5, 6, 7000), 7),
        (time(4, 5, 6, 999), 0),
        (None, None),
    ],
    ids=[
        "datetime - with milliseconds",
        "datetime - without full milliseconds",
        "time - with milliseconds",
        "time - without full milliseconds",
        "None",
    ],
)
def test_should_extract_millisecond(
    value: datetime | time | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.millisecond(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
