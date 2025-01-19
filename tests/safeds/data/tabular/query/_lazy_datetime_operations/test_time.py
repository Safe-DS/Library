from datetime import UTC, datetime, time

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 2, 3, 4, 5, 6, 7, tzinfo=UTC), time(4, 5, 6, 7)),
        (None, None),
    ],
    ids=[
        "datetime",
        "None",
    ],
)
def test_should_extract_time(
    value: datetime | None,
    expected: time | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.time(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
