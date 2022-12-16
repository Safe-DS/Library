import pandas as pd
from _pytest.python_api import raises
from safe_ds.data import IntColumnType, Row, StringColumnType, Table, TableSchema
from safe_ds.exceptions import SchemaMismatchError


def test_add_row_valid() -> None:
    table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    row = Row(pd.Series(data=[5, 6]), table1.schema)
    table1 = table1.add_row(row)
    assert table1.count_rows() == 4
    assert table1.get_row(3) == row
    assert table1.schema == row.schema


def test_add_row_invalid() -> None:
    with raises(SchemaMismatchError):
        table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
        row = Row(
            pd.Series(data=[5, "Hallo"]),
            TableSchema(["col1", "col2"], [IntColumnType(), StringColumnType()]),
        )
        table1 = table1.add_row(row)
