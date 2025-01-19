from datetime import UTC, date, datetime

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(1, 1, 1, tzinfo=UTC), 1),
        (datetime(100, 12, 31, tzinfo=UTC), 1),
        (datetime(101, 1, 1, tzinfo=UTC), 2),
        (date(1, 1, 1), 1),
        (date(100, 12, 31), 1),
        (date(101, 1, 1), 2),
        (None, None),
    ],
    ids=[
        "datetime - first day of first century",
        "datetime - last day of first century",
        "datetime - first day of second century",
        "date - first day of first century",
        "date - last day of first century",
        "date - first day of second century",
        "None",
    ],
)
def test_should_extract_century(
    value: datetime | date | None,
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.century(),
        expected,
        type_if_none=ColumnType.datetime(),
    )
