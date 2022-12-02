import numpy as np
from safe_ds.data import ColumnType, Table


def test_get_type_of_column():
    table = Table.from_json("tests/resources/test_schema_table.json")
    table_column_type = table.schema.get_type_of_column("A")
    assert table_column_type == ColumnType.from_numpy_dtype(np.dtype("int64"))
