import polars as pl
import pytest

from safeds.data.tabular.containers._lazy_cell import _LazyCell
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (3, 3, 0),
        (3, 1.5, 0.0),
        (1.5, 3, 1.5),
        (1.5, 1.5, 0.0),
        (None, 3, None),
        (3, None, None),
    ],
    ids=[
        "int - int",
        "int - float",
        "float - int",
        "float - float",
        "left is None",
        "right is None",
    ],
)
class TestShouldComputeModulus:
    def test_dunder_method(self, value1: float | None, value2: float | None, expected: float | None) -> None:
        assert_cell_operation_works(value1, lambda cell: cell % value2, expected)

    def test_dunder_method_wrapped_in_cell(
        self,
        value1: float | None,
        value2: float | None,
        expected: float | None,
    ) -> None:
        assert_cell_operation_works(value1, lambda cell: cell % _LazyCell(pl.lit(value2)), expected)

    def test_dunder_method_inverted_order(
        self,
        value1: float | None,
        value2: float | None,
        expected: float | None,
    ) -> None:
        assert_cell_operation_works(value2, lambda cell: value1 % cell, expected)

    def test_dunder_method_inverted_order_wrapped_in_cell(
        self,
        value1: float | None,
        value2: float | None,
        expected: float | None,
    ) -> None:
        assert_cell_operation_works(value2, lambda cell: _LazyCell(pl.lit(value1)) % cell, expected)

    def test_named_method(self, value1: float | None, value2: float | None, expected: float | None) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.mod(value2), expected)

    def test_named_method_wrapped_in_cell(
        self,
        value1: float | None,
        value2: float | None,
        expected: float | None,
    ) -> None:
        assert_cell_operation_works(value1, lambda cell: cell.mod(_LazyCell(pl.lit(value2))), expected)
