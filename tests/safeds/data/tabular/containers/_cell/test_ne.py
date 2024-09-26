import pytest

import polars as pl

from safeds.data.tabular.containers._lazy_cell import _LazyCell
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (3, 3, False),
        (3, 1.5, True),
        (1.5, 3, True),
        (1.5, 1.5, False),
    ],
    ids=[
        "int - int",
        "int - float",
        "float - int",
        "float - float",
    ],
)
class TestShouldComputeNegatedEquality:
    def test_dunder_method(self, value1: float, value2: float, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell != value2, expected)

    def test_dunder_method_wrapped_in_cell(self, value1: float, value2: float, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell != _LazyCell(pl.lit(value2)), expected)

    def test_dunder_method_inverted_order(self, value1: float, value2: float, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: value2 != cell, expected)  # type: ignore[arg-type,return-value]

    def test_dunder_method_inverted_order_wrapped_in_cell(self, value1: float, value2: float, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: _LazyCell(pl.lit(value2)) != cell, expected)  # type: ignore[arg-type,return-value]

    def test_named_method(self, value1: float, value2: float, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.neq(value2), expected)

    def test_named_method_wrapped_in_cell(self, value1: float, value2: float, expected: bool) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.neq(_LazyCell(pl.lit(value2))), expected)
