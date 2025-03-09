from datetime import time

import pytest

from safeds._typing import _ConvertibleToIntCell
from safeds.data.tabular.containers import Cell, Column
from safeds.exceptions import LazyComputationError
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("hour", "minute", "second", "microsecond", "expected"),
    [
        (1, 2, 3, 4, time(1, 2, 3, 4)),
        (Cell.constant(1), Cell.constant(2), Cell.constant(3), Cell.constant(4), time(1, 2, 3, 4)),
        # with None components
        (None, 2, 3, 4, None),
        (1, None, 3, 4, None),
        (1, 2, None, 4, None),
        (1, 2, 3, None, None),
    ],
    ids=[
        "int components",
        "cell components",
        "hour is None",
        "minute is None",
        "second is None",
        "microsecond is None",
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


@pytest.mark.parametrize(
    ("hour", "minute", "second", "microsecond"),
    [
        # invalid hour
        (-1, 2, 3, 4),
        (24, 2, 3, 4),
        # invalid minute
        (1, -1, 3, 4),
        (1, 60, 3, 4),
        # invalid second
        (1, 2, -1, 4),
        (1, 2, 60, 4),
        # invalid microsecond
        (1, 2, 3, -1),
        pytest.param(
            1, 2, 3, 1_000_000, marks=pytest.mark.xfail(reason="https://github.com/pola-rs/polars/issues/21664"),
        ),
    ],
    ids=[
        "hour is too low",
        "hour is too high",
        "minute is too low",
        "minute is too high",
        "second is too low",
        "second is too high",
        "microsecond is too low",
        "microsecond is too high",
    ],
)
def test_should_raise_for_invalid_components(
    hour: _ConvertibleToIntCell,
    minute: _ConvertibleToIntCell,
    second: _ConvertibleToIntCell,
    microsecond: _ConvertibleToIntCell,
) -> None:
    column = Column("col1", [None])
    with pytest.raises(LazyComputationError):
        column.transform(
            lambda _: Cell.time(hour, minute, second, microsecond=microsecond),
        ).get_value(0)
