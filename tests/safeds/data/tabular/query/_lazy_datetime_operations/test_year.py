from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1999, 1, 1, tzinfo=UTC), 1999),
        (date(1999, 1, 1), 1999),
        (None, None),
    ],
    ids=[
        "datetime",
        "date",
        "None",
    ],
)
def test_should_return_unix_timestamp(
    value: datetime | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.year(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
