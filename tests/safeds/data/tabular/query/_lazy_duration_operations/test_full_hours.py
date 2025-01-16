from datetime import timedelta

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(hours=1), 1),
        (timedelta(hours=1, minutes=30), 1),
        (timedelta(hours=-1), -1),
        (timedelta(hours=-1, minutes=-30), -1),
        (timedelta(hours=1, minutes=-30), 0),
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
def test_should_return_full_hours(value: timedelta | None, expected: int | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.dur.full_hours(), expected, type_if_none=ColumnType.duration())
