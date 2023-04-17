from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from safeds.data.tabular.containers import Column

if TYPE_CHECKING:
    from typing import Any


@pytest.mark.parametrize(
    ("values", "unique_values"),
    [([1, 1, 2, 3], [1, 2, 3]), (["a", "b", "b", "c"], ["a", "b", "c"]), ([], [])],
)
def test_get_unique_values(values: list[Any], unique_values: list[Any]) -> None:
    column: Column = Column("", values)
    extracted_unique_values = column.get_unique_values()

    assert extracted_unique_values == unique_values
