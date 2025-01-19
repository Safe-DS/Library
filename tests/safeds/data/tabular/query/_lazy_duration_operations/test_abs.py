from datetime import timedelta

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(days=1), timedelta(days=1)),
        (timedelta(days=1, hours=12), timedelta(days=1, hours=12)),
        (timedelta(days=-1), timedelta(days=1)),
        (timedelta(days=-1, hours=-12), timedelta(days=1, hours=12)),
        (timedelta(days=1, hours=-12), timedelta(hours=12)),
        (timedelta(days=-1, hours=12), timedelta(hours=12)),
        (None, None),
    ],
    ids=[
        "positive days",
        "positive days and hours",
        "negative days",
        "negative days and hours",
        "positive days, negative hours",
        "negative days, positive hours",
        "None",
    ],
)
def test_should_return_absolute_duration(value: timedelta | None, expected: timedelta | None) -> None:
    assert_cell_operation_works(value, lambda cell: cell.dur.abs(), expected, type_if_none=ColumnType.duration())
