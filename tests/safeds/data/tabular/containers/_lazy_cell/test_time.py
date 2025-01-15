from datetime import time

import pytest
from helpers import assert_cell_operation_works

from safeds._typing import _ConvertibleToIntCell
from safeds.data.tabular.containers import Cell


@pytest.mark.parametrize(
    ("hour", "minute", "second", "microsecond", "expected"),
    [
        (1, 2, 3, 4, time(1, 2, 3, 4)),
        (Cell.constant(1), Cell.constant(2), Cell.constant(3), Cell.constant(4), time(1, 2, 3, 4)),
        # invalid hour
        (None, 2, 3, 4, None),
        (-1, 2, 3, 4, None),
        (24, 2, 3, 4, None),
        # invalid minute
        (1, None, 3, 4, None),
        (1, -1, 3, 4, None),
        (1, 60, 3, 4, None),
        # invalid second
        (1, 2, None, 4, None),
        (1, 2, -1, 4, None),
        (1, 2, 60, 4, None),
        # invalid microsecond
        (1, 2, 3, None, None),
        (1, 2, 3, -1, None),
        (1, 2, 3, 1_000_000, None),
    ],
    ids=[
        "int components",
        "cell components",
        "hour is None",
        "hour is too low",
        "hour is too high",
        "minute is None",
        "minute is too low",
        "minute is too high",
        "second is None",
        "second is too low",
        "second is too high",
        "microsecond is None",
        "microsecond is too low",
        "microsecond is too high",
    ],
)
def test_should_return_time(
    hour: _ConvertibleToIntCell,
    minute: _ConvertibleToIntCell,
    second: _ConvertibleToIntCell,
    microsecond: _ConvertibleToIntCell,
    expected: time,
) -> None:
    assert_cell_operation_works(
        None,
        lambda _: Cell.time(hour, minute, second, microsecond=microsecond),
        expected,
    )
