from datetime import timedelta

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (timedelta(microseconds=1), 1),
        (timedelta(microseconds=-1), -1),
        (timedelta(milliseconds=1, microseconds=-500), 500),
        (None, None),
    ],
    ids=[
        "positive, exact",
        "negative, exact",
        "mixed",
        "None",
    ],
)
def test_should_return_full_microseconds(value: timedelta | None, expected: int | None) -> None:
    assert_cell_operation_works(
        value,
        lambda cell: cell.dur.full_microseconds(),
        expected,
        type_if_none=ColumnType.duration(),
    )
