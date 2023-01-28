import pandas as pd
from safeds.data import Column, IntColumnType, StringColumnType, Table


def test_add_columns_valid() -> None:
    table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    col3 = Column(pd.Series(data=[0, -1, -2]), "col3")
    col4 = Column(pd.Series(data=["a", "b", "c"]), "col4")
    table1 = table1.add_columns([col3, col4])
    assert table1.count_columns() == 4
    assert table1.get_column("col3") == col3
    assert table1.get_column("col4") == col4
    assert isinstance(table1.schema.get_type_of_column("col1"), IntColumnType)
    assert isinstance(table1.schema.get_type_of_column("col2"), IntColumnType)
    assert isinstance(table1.schema.get_type_of_column("col3"), IntColumnType)
    assert isinstance(table1.schema.get_type_of_column("col4"), StringColumnType)


def test_add_columns_table_valid() -> None:
    table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    col3 = Column(pd.Series(data=[0, -1, -2]), "col3")
    col4 = Column(pd.Series(data=["a", "b", "c"]), "col4")
    table2 = Table.from_columns([col3, col4])
    table1 = table1.add_columns(table2)
    assert table1.count_columns() == 4
    assert table1.get_column("col3") == col3
    assert table1.get_column("col4") == col4
    assert isinstance(table1.schema.get_type_of_column("col1"), IntColumnType)
    assert isinstance(table1.schema.get_type_of_column("col2"), IntColumnType)
    assert isinstance(table1.schema.get_type_of_column("col3"), IntColumnType)
    assert isinstance(table1.schema.get_type_of_column("col4"), StringColumnType)
