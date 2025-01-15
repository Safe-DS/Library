from typing import Any

import pytest

from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, True),
        (True, False),
        (None, None),
        (0, True),
        (1, False),
    ],
    ids=[
        "False",
        "True",
        "None",
        "falsy int",
        "truthy int",
    ],
)
class TestShouldInvertValueOfCell:
    def test_dunder_method(self, value: Any, expected: bool | None) -> None:
        assert_cell_operation_works(value, lambda cell: ~cell, expected, type_if_none=ColumnType.boolean())

    def test_named_method(self, value: Any, expected: bool | None) -> None:
        assert_cell_operation_works(value, lambda cell: cell.not_(), expected, type_if_none=ColumnType.boolean())
