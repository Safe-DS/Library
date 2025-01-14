import datetime
from typing import Any

import pytest

from safeds.data.tabular.containers import Column


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            [],
            1,
        ),
        (
            [None],
            1,
        ),
        (
            [1, 2, 1, 1, None],
            3 / 4,
        ),
        (
            [datetime.time(1, 2, 3), datetime.time(4, 5, 6), None],
            1 / 2,
        ),
        (
            ["a", "b", "a", "b", None],
            2 / 4,
        ),
        (
            [True, False, True, True, None],
            3 / 4,
        ),
    ],
    ids=[
        "empty",
        "null column",
        "numeric column",
        "temporal column",
        "string column",
        "boolean column",  # Failed due to a polars error in previous implementation
    ],
)
def test_should_return_stability_of_column(values: list[Any], expected: float) -> None:
    column = Column("col1", values)
    assert column.stability() == expected
