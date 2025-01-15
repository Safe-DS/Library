from typing import Any

import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._cell import Cell

_none_cell = Cell.constant(None)


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ([], None),
        ([_none_cell], None),
        ([_none_cell, Cell.constant(1)], 1),
        ([Cell.constant(1), _none_cell, Cell.constant(2)], 1),
    ],
    ids=[
        "empty",
        "all None",
        "one not None",
        "multiple not None",
    ],
)
def test_should_return_first_non_none_value(cells: list[Cell], expected: Any) -> None:
    table = Table({"col1": [1]})
    actual = table.add_computed_column("col2", lambda _: Cell.first_not_none(cells))
    assert actual.get_column("col2").get_value(0) == expected
