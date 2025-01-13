from typing import Any

import polars as pl
import pytest

from safeds.data.tabular.containers._lazy_cell import _LazyCell
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (0, False, False),
        (0, True, True),
        (1, False, True),
        (1, True, False),
    ],
    ids=[
        "false - false",
        "false - true",
        "true - false",
        "true - true",
        "falsy int - false",
        "falsy int - true",
        "truthy int - false",
        "truthy int - true",
    ],
)
class TestShouldComputeExclusiveOr:
    def test_dunder_method(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell ^ value2, expected)

    def test_dunder_method_wrapped_in_cell(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell ^ _LazyCell(pl.lit(value2)), expected)

    def test_dunder_method_inverted_order(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value2, lambda cell: value1 ^ cell, expected)

    def test_dunder_method_inverted_order_wrapped_in_cell(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value2, lambda cell: _LazyCell(pl.lit(value1)) ^ cell, expected)

    def test_named_method(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.xor(value2), expected)

    def test_named_method_wrapped_in_cell(self, value1: Any, value2: bool, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.xor(_LazyCell(pl.lit(value2))), expected)
