import numpy as np
from safe_ds.data import Table


def test_get_type_of_column():
    table = Table.from_json("tests/resources/test_schema_table.json")

    table_column_type = table.schema.get_type_of_column("A")

    assert table_column_type is np.dtype("int64")
