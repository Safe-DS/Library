import pytest
from safeds.data.tabular.containers import Table, TaggedTable, TimeSeries


def assert_that_tables_are_close(table1: Table, table2: Table) -> None:
    """
    Assert that two tables are almost equal.

    Parameters
    ----------
    table1: Table
        The first table.
    table2: Table
        The table to compare the first table to.
    """
    assert table1.schema == table2.schema
    for column_name in table1.column_names:
        assert table1.get_column(column_name).type == table2.get_column(column_name).type
        assert table1.get_column(column_name).type.is_numeric()
        assert table2.get_column(column_name).type.is_numeric()
        for i in range(table1.number_of_rows):
            entry_1 = table1.get_column(column_name).get_value(i)
            entry_2 = table2.get_column(column_name).get_value(i)
            assert entry_1 == pytest.approx(entry_2)


def assert_that_tagged_tables_are_equal(table1: TaggedTable, table2: TaggedTable) -> None:
    """
    Assert that two tagged tables are equal.

    Parameters
    ----------
    table1: TaggedTable
        The first table.
    table2: TaggedTable
        The table to compare the first table to.
    """
    assert table1.schema == table2.schema
    assert table1.features == table2.features
    assert table1.target == table2.target
    assert table1 == table2

def assert_that_time_series_are_equal(table1: TimeSeries, table2: TimeSeries) -> None:
    """
    Assert that two time series are equal.

    Parameters
    ----------
    table1: TimeSeries
        The first timeseries.
    table2: TimeSeries
        The timeseries to compare the first timeseries to.
    """
    assert table1.schema == table2.schema
    assert table1.features == table2.features
    assert table1.target == table2.target
    assert table1.time == table2.time
    assert table1 == table2