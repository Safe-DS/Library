import numpy as np
from safe_ds.data import Table, TableSchema


def test_table_equals_valid():
    table = Table.from_json("tests/resources/test_schema_table.json")
    schema_expected = TableSchema(["A", "B"], [np.dtype("int64"), np.dtype("int64")])

    assert table.schema == schema_expected


def test_table_equals_invalid():
    table = Table.from_json("tests/resources/test_schema_table.json")
    schema_not_expected = TableSchema(["A", "C"], [np.dtype("f8"), np.dtype("int64")])

    assert table.schema != schema_not_expected
