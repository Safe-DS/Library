from datetime import UTC, datetime
from typing import Literal

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (datetime(1970, 1, 1, tzinfo=UTC), "s", 0),
        (datetime(1969, 12, 31, tzinfo=UTC), "s", -86400),
        (datetime(1970, 1, 2, tzinfo=UTC), "s", 86400),
        (datetime(1970, 1, 2, tzinfo=UTC), "ms", 86400000),
        (datetime(1970, 1, 2, tzinfo=UTC), "us", 86400000000),
        (None, "s", None),
    ],
    ids=[
        "epoch",
        "one day before epoch",
        "one day after epoch (seconds)",
        "one day after epoch (milliseconds)",
        "one day after epoch (microseconds)",
        "None",
    ],
)
def test_should_return_unix_timestamp(
    value: datetime | None,
    unit: Literal["s", "ms", "us"],
    expected: int | None,
) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dt.unix_timestamp(unit=unit),
        expected,
        type_if_none=ColumnType.datetime(),
    )
