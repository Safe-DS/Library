from datetime import timedelta

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(weeks=1), 1),
        (timedelta(weeks=1, days=3), 1),
        (timedelta(weeks=-1), -1),
        (timedelta(weeks=-1, days=-3), -1),
        (timedelta(weeks=1, days=-3), 0),
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
def test_should_return_full_weeks(value: timedelta | None, expected: int | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.dur.full_weeks(), expected, type_if_none=ColumnType.duration())
