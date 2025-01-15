from typing import Any

import pytest
from helpers import assert_cell_operation_works

from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("value", "type_", "expected"),
    [
        (1, ColumnType.string(), "1"),
        ("1", ColumnType.int64(), 1),
        (None, ColumnType.int64(), None),
    ],
    ids=[
        "int64 to string",
        "string to int64",
        "None to int64",
    ],
)
def test_should_cast_values_to_requested_type(value: Any, type_: ColumnType, expected: Any) -> None:
    assert_cell_operation_works(value, lambda cell: cell.cast(type_), expected)
