from datetime import timedelta

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(seconds=1), 1),
        (timedelta(seconds=1, milliseconds=500), 1),
        (timedelta(seconds=-1), -1),
        (timedelta(seconds=-1, milliseconds=-500), -1),
        (timedelta(seconds=1, milliseconds=-500), 0),
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
def test_should_return_full_seconds(value: timedelta | None, expected: int | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dur.full_seconds(),
        expected,
        type_if_none=ColumnType.duration(),
    )
