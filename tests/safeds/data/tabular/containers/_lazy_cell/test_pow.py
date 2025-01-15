import polars as pl
import pytest

from safeds.data.tabular.containers._lazy_cell import _LazyCell
from safeds.data.tabular.typing import ColumnType
from tests.helpers import assert_cell_operation_works


@pytest.mark.parametrize(
    ("value1", "value2", "expected"),
    [
        (3, 2, 9),
        (4, 0.5, 2.0),
        (1.5, 2, 2.25),
        (2.25, 0.5, 1.5),
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
class TestShouldComputePower:
    def test_dunder_method(self, value1: float | None, value2: float | None, expected: float | None) -> None:
        if value2 is None:
            pytest.skip("polars does not support null exponents.")

        assert_cell_operation_works(value1, lambda cell: cell**value2, expected, type_if_none=ColumnType.float64())

    def test_dunder_method_wrapped_in_cell(
        self,
        value1: float | None,
        value2: float | None,
        expected: float | None,
    ) -> None:
        assert_cell_operation_works(
            value1,
            lambda cell: cell ** _LazyCell(pl.lit(value2, dtype=pl.Float64())),
            expected,
            type_if_none=ColumnType.float64(),
        )

    def test_dunder_method_inverted_order(
        self,
        value1: float | None,
        value2: float | None,
        expected: float | None,
    ) -> None:
        if value1 is None:
            pytest.skip("polars does not support null base.")

        assert_cell_operation_works(value2, lambda cell: value1**cell, expected, type_if_none=ColumnType.float64())

    def test_dunder_method_inverted_order_wrapped_in_cell(
        self,
        value1: float | None,
        value2: float | None,
        expected: float | None,
    ) -> None:
        assert_cell_operation_works(
            value2,
            lambda cell: _LazyCell(pl.lit(value1, dtype=pl.Float64())) ** cell,
            expected,
            type_if_none=ColumnType.float64(),
        )

    def test_named_method(self, value1: float | None, value2: float | None, expected: float | None) -> None:
        if value2 is None:
            pytest.skip("polars does not support null exponents.")

        assert_cell_operation_works(value1, lambda cell: cell.pow(value2), expected, type_if_none=ColumnType.float64())

    def test_named_method_wrapped_in_cell(
        self,
        value1: float | None,
        value2: float | None,
        expected: float | None,
    ) -> None:
        assert_cell_operation_works(
            value1,
            lambda cell: cell.pow(_LazyCell(pl.lit(value2, dtype=pl.Float64()))),
            expected,
            type_if_none=ColumnType.float64(),
        )
