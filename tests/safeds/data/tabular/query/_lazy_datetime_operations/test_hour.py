from datetime import UTC, datetime, time

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 2, 3, 4, 5, 6, 7, tzinfo=UTC), 4),
        (time(4, 5, 6, 7), 4),
        (None, None),
    ],
    ids=[
        "datetime",
        "time",
        "None",
    ],
)
def test_should_extract_hour(
    value: datetime | time | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.hour(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
