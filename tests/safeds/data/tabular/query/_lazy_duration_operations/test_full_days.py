from datetime import timedelta

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(days=1), 1),
        (timedelta(days=1, hours=12), 1),
        (timedelta(days=-1), -1),
        (timedelta(days=-1, hours=-12), -1),
        (timedelta(days=1, hours=-12), 0),
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
def test_should_return_full_days(value: timedelta | None, expected: int | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.dur.full_days(), expected, type_if_none=ColumnType.duration())
