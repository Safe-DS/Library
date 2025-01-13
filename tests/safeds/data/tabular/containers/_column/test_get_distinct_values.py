from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from safeds.data.tabular.containers import Column

if TYPE_CHECKING:
    from typing import Any


@pytest.mark.parametrize(
    ("values", "ignore_missing_values", "expected"),
    [
        ([], True, []),
        ([1, 2, 3], True, [1, 2, 3]),
        ([1, 2, 1], True, [1, 2]),
        ([1, 2, 3, None], True, [1, 2, 3]),
        ([1, 2, 3, None], False, [1, 2, 3, None]),
    ],
    ids=[
        "empty",
        "no duplicates",
        "some duplicate",
        "with missing values (ignored)",
        "with missing values (not ignored)",
    ],
)
def test_should_get_distinct_values(values: list[Any], ignore_missing_values: bool, expected: list[Any]) -> None:
    column: Column = Column("col1", values)
    assert column.get_distinct_values(ignore_missing_values=ignore_missing_values) == expected
