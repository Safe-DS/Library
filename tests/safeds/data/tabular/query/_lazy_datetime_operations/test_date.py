from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 2, 3, tzinfo=UTC), date(1, 2, 3)),
        (None, None),
    ],
    ids=[
        "datetime",
        "None",
    ],
)
def test_should_extract_date(
    value: datetime | None,
    expected: date | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.date(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
