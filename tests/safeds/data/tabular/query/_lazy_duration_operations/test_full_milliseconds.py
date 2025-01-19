from datetime import timedelta

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(milliseconds=1), 1),
        (timedelta(milliseconds=1, microseconds=500), 1),
        (timedelta(milliseconds=-1), -1),
        (timedelta(milliseconds=-1, microseconds=-500), -1),
        (timedelta(milliseconds=1, microseconds=-500), 0),
        (None, None),
    ],
    ids=[
        "positive, exact",
        "positive, rounded",
        "negative, exact",
        "negative, rounded",
        "mixed",
        "None",
    ],
)
def test_should_return_full_milliseconds(value: timedelta | None, expected: int | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dur.full_milliseconds(),
        expected,
        type_if_none=ColumnType.duration(),
    )
