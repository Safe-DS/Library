from typing import Any

import pytest

from safeds.data.tabular.containers import Cell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "type_", "expected"),
    [
        (None, None, None),
        (1, None, 1),
        (1, ColumnType.string(), "1"),
    ],
    ids=[
        "None",
        "int",
        "with explicit type",
    ],
)
def test_should_return_constant_value(value: Any, type_: ColumnType | None, expected: Any) -> None:
    assert_cell_operation_works(None, lambda _: Cell.constant(value, type=type_), expected)
