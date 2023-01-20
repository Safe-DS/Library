import pandas as pd
from safe_ds.data import Row, Table


def test_add_rows_valid() -> None:
    table1 = Table(pd.DataFrame(data={"col1": ["a", "b", "c"], "col2": [1, 2, 4]}))
    row1 = Row(pd.Series(data=["d", 6]), table1.schema)
    row2 = Row(pd.Series(data=["e", 8]), table1.schema)
    table1 = table1.add_rows([row1, row2])
    assert table1.count_rows() == 5
    assert table1.get_row(3) == row1
    assert table1.get_row(4) == row2
    assert table1.schema == row1.schema
    assert table1.schema == row2.schema


def test_add_rows_table_valid() -> None:
    table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    row1 = Row(pd.Series(data=[5, 6]), table1.schema)
    row2 = Row(pd.Series(data=[7, 8]), table1.schema)
    table2 = Table.from_rows([row1, row2])
    table1 = table1.add_rows(table2)
    assert table1.count_rows() == 5
    assert table1.get_row(3) == row1
    assert table1.get_row(4) == row2
    assert table1.schema == row1.schema
    assert table1.schema == row2.schema
