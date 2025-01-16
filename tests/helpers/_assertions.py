from collections.abc import Callable
from typing import Any

from polars.testing import assert_frame_equal

from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Cell, Column, Row, Table
from safeds.data.tabular.typing import ColumnType


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


def assert_tabular_datasets_are_equal(table1: TabularDataset, table2: TabularDataset) -> None:
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
    value: Any,
    transformer: Callable[[Cell], Cell],
    expected: Any,
    *,
    type_if_none: ColumnType | None = None,
) -> None:
    """
    Assert that a cell operation works as expected.

    Parameters
    ----------
    value:
        The value in the input cell.
    transformer:
        The transformer to apply to the cells.
    expected:
        The expected value of the transformed cell.
    type_if_none:
        The type of the column if the value is `None`.
    """
    type_ = type_if_none if value is None else None
    column = Column("a", [value], type=type_)
    transformed_column = column.transform(transformer)
    actual = transformed_column[0]
    assert actual == expected, f"Expected {expected}, but got {actual}."


def assert_row_operation_works(
    table: Table,
    computer: Callable[[Row], Cell],
    expected: list[Any],
) -> None:
    """
    Assert that a row operation works as expected.

    Parameters
    ----------
    table:
        The input table.
    computer:
        The function that computes the new column.
    expected:
        The expected values of the computed column.
    """
    column_name = _find_free_column_name(table, "computed")

    new_table = table.add_computed_column(column_name, computer)
    actual = list(new_table.get_column(column_name))
    assert actual == expected


def _find_free_column_name(table: Table, prefix: str) -> str:
    """
    Find a free column name in the table.

    Parameters
    ----------
    table:
        The table to search for a free column name.
    prefix:
        The prefix to use for the column name.

    Returns
    -------
    free_name:
        A free column name.
    """
    column_name = prefix

    while column_name in table.column_names:
        column_name += "_"

    return column_name
