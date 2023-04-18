from typing import Any

import pytest
from safeds.data.tabular.containers import Column
from safeds.data.tabular.exceptions import ColumnSizeError


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3, 4, None], 1 / 4),
        ([1, 1, 3, "abc", None], 2 / 4),
        (["b", "a", "abc", "abc", "abc"], 3 / 5),
    ],
)
def test_stability(values: list[Any], expected: float) -> None:
    column = Column("A", values)
    assert column.stability() == expected


def test_stability_error() -> None:
    column = Column("A", [])
    with pytest.raises(ColumnSizeError):
        column.stability()
