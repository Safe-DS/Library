from collections.abc import Callable
from typing import Any

from polars.testing import assert_frame_equal
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Cell, Column, Table


def assert_tables_equal(table1: Table, table2: Table) -> None:
    """
    Assert that two tables are almost equal.

    Parameters
    ----------
    table1:
        The first table.
    table2:
        The table to compare the first table to.
    """
    assert_frame_equal(table1._data_frame, table2._data_frame)


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
    input_data: list[Any],
    transformer: Callable[[Cell], Cell],
    expected_data: list[Any],
) -> None:
    """
    Assert that a cell operation works as expected.

    Parameters
    ----------
    input_data:
        The input data.
    transformer:
        The transformer to apply to the cells.
    expected_data:
        The expected data.
    """
    column = Column("A", input_data)
    transformed_column = column.transform(transformer)
    assert transformed_column == Column("A", expected_data)
