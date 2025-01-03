import polars as pl
import pytest
from safeds.data.tabular.containers._lazy_cell import _LazyCell

from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (3, 2, 9),
        (4, 0.5, 2.0),
        (1.5, 2, 2.25),
        (2.25, 0.5, 1.5),
    ],
    ids=[
        "int - int",
        "int - float",
        "float - int",
        "float - float",
    ],
)
class TestShouldComputePower:
    def test_dunder_method(self, value1: float, value2: float, expected: float) -> None:
        assert_cell_operation_works(value1, lambda cell: cell**value2, expected)

    def test_dunder_method_wrapped_in_cell(self, value1: float, value2: float, expected: float) -> None:
        assert_cell_operation_works(value1, lambda cell: cell ** _LazyCell(pl.lit(value2)), expected)

    def test_dunder_method_inverted_order(self, value1: float, value2: float, expected: float) -> None:
        assert_cell_operation_works(value2, lambda cell: value1**cell, expected)

    def test_dunder_method_inverted_order_wrapped_in_cell(self, value1: float, value2: float, expected: float) -> None:
        assert_cell_operation_works(value2, lambda cell: _LazyCell(pl.lit(value1)) ** cell, expected)

    def test_named_method(self, value1: float, value2: float, expected: float) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.pow(value2), expected)

    def test_named_method_wrapped_in_cell(self, value1: float, value2: float, expected: float) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.pow(_LazyCell(pl.lit(value2))), expected)
