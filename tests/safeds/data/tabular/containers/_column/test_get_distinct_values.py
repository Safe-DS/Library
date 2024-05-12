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
        ([None, None, None], [None]),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 1, 2, 3], [1, 2, 3]),
        (["a", "b", "b", "c"], ["a", "b", "c"]),
    ],
    ids=[
        "empty",
        "only missing",
        "no duplicates",
        "integer duplicates",
        "string duplicates",
    ],
)
def test_should_list_unique_values(values: list[Any], expected: list[Any]) -> None:
    column: Column = Column("", values)
    assert column.get_distinct_values() == expected
