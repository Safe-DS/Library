from collections.abc import Callable
from typing import Any

from polars.testing import assert_frame_equal

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Cell, Column, Table


def assert_tables_are_equal(
    actual: Table,
    expected: Table,
    *,
    ignore_column_order: bool = False,
    ignore_row_order: bool = False,
    ignore_types: bool = False,
    ignore_float_imprecision: bool = True,
) -> None:
    """
    Assert that two tables are equal.

    Parameters
    ----------
    actual:
        The actual table.
    expected:
        The expected table.
    ignore_column_order:
        Ignore the column order when True.
    ignore_row_order:
        Ignore the column order when True.
    ignore_types:
        Ignore differing data types.
    ignore_float_imprecision:
        If False, check if floating point values match EXACTLY.
    """
    assert_frame_equal(
        actual._data_frame,
        expected._data_frame,
        check_row_order=not ignore_row_order,
        check_column_order=not ignore_column_order,
        check_dtypes=not ignore_types,
        check_exact=not ignore_float_imprecision,
    )


def assert_that_tabular_datasets_are_equal(table1: TabularDataset, table2: TabularDataset) -> None:
    """
    Assert that two tabular datasets are equal.

    Parameters
    ----------
    table1:
        The first table.
    table2:
        The table to compare the first table to.
    """
    assert table1._table.schema == table2._table.schema
    assert table1.features == table2.features
    assert table1.target == table2.target
    assert table1 == table2


def assert_cell_operation_works(
    input_value: Any,
    transformer: Callable[[Cell], Cell],
    expected_value: Any,
) -> None:
    """
    Assert that a cell operation works as expected.

    Parameters
    ----------
    input_value:
        The value in the input cell.
    transformer:
        The transformer to apply to the cells.
    expected_value:
        The expected value of the transformed cell.
    """
    column = Column("A", [input_value])
    transformed_column = column.transform(transformer)
    assert transformed_column == Column("A", [expected_value]), f"Expected: {expected_value}\nGot: {transformed_column}"


def assert_row_operation_works(
    input_value: Any,
    transformer: Callable[[Table], Table],
    expected_value: Any,
) -> None:
    """
    Assert that a row operation works as expected.

    Parameters
    ----------
    input_value:
        The value in the input row.
    transformer:
        The transformer to apply to the rows.
    expected_value:
        The expected value of the transformed row.
    """
    table = Table(input_value)
    transformed_table = transformer(table)
    assert transformed_table == Table(expected_value), f"Expected: {expected_value}\nGot: {transformed_table}"
