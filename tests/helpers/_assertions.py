from polars.testing import assert_frame_equal
from safeds.data.labeled.containers import TabularDataset
from safeds.data.tabular.containers import Table


def assert_tables_equal(table1: Table, table2: Table) -> None:
    """
    Assert that two tables are almost equal.

    Parameters
    ----------
    table1: Table
        The first table.
    table2: Table
        The table to compare the first table to.
    """
    assert_frame_equal(table1._data_frame, table2._data_frame)


def assert_that_tabular_datasets_are_equal(table1: TabularDataset, table2: TabularDataset) -> None:
    """
    Assert that two tabular datasets are equal.

    Parameters
    ----------
    table1: TabularDataset
        The first table.
    table2: TabularDataset
        The table to compare the first table to.
    """
    assert table1._table.schema == table2._table.schema
    assert table1.features == table2.features
    assert table1.target == table2.target
    assert table1 == table2
