from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (False, False, False),
        (False, True, True),
        (False, None, None),
        (True, False, True),
        (True, True, False),
        (True, None, None),
        (None, False, None),
        (None, True, None),
        (None, None, None),
        (0, False, False),
        (0, True, True),
        (1, False, True),
        (1, True, False),
    ],
    ids=[
        "False - False",
        "False - True",
        "False - None",
        "True - False",
        "True - True",
        "True - None",
        "None - False",
        "None - True",
        "None - None",
        "falsy int - False",
        "falsy int - True",
        "truthy int - False",
        "truthy int - True",
    ],
)
class TestShouldComputeExclusiveOr:
    def test_dunder_method(self, value1: Any, value2: bool | None, expected: bool | None) -> None:
        assert_cell_operation_works(value1, lambda cell: cell ^ value2, expected, type_if_none=ColumnType.boolean())

    def test_dunder_method_wrapped_in_cell(self, value1: Any, value2: bool | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value1,
            lambda cell: cell ^ _LazyCell(pl.lit(value2)),
            expected,
            type_if_none=ColumnType.boolean(),
        )

    def test_dunder_method_inverted_order(self, value1: Any, value2: bool | None, expected: bool | None) -> None:
        assert_cell_operation_works(value2, lambda cell: value1 ^ cell, expected, type_if_none=ColumnType.boolean())

    def test_dunder_method_inverted_order_wrapped_in_cell(
        self,
        value1: Any,
        value2: bool | None,
        expected: bool | None,
    ) -> None:
        assert_cell_operation_works(
            value2,
            lambda cell: _LazyCell(pl.lit(value1)) ^ cell,
            expected,
            type_if_none=ColumnType.boolean(),
        )

    def test_named_method(self, value1: Any, value2: bool | None, expected: bool | None) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.xor(value2), expected, type_if_none=ColumnType.boolean())

    def test_named_method_wrapped_in_cell(self, value1: Any, value2: bool | None, expected: bool | None) -> None:
        assert_cell_operation_works(
            value1,
            lambda cell: cell.xor(_LazyCell(pl.lit(value2))),
            expected,
            type_if_none=ColumnType.boolean(),
        )
