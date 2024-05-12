from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from safeds.data.tabular.containers import Column

if TYPE_CHECKING:
    from typing import Any


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], []),
        ([1, None, None], [1]),
        ([None, None, None], []),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 1, 2, 3], [1, 2, 3]),
        (["a", "b", "b", "c"], ["a", "b", "c"]),
    ],
    ids=[
        "empty",
        "some missing values",
        "only missing values",
        "no duplicates",
        "integer duplicates",
        "string duplicates",
    ],
)
def test_should_get_unique_values_ignoring_missing_values(values: list[Any], expected: list[Any]) -> None:
    column: Column = Column("", values)
    assert column.get_distinct_values() == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, None, None], [None, 1]),
        ([None, None, None], [None]),
    ],
    ids=[
        "some missing values",
        "only missing values",
    ],
)
def test_should_get_unique_values_including_missing_values_if_requested(values: list[Any], expected: list[Any]) -> None:
    column: Column = Column("", values)
    assert column.get_distinct_values(ignore_missing_values=False) == expected
