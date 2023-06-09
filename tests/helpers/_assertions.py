import pytest
from safeds.data.tabular.containers import Table


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
