import numpy as np
from safe_ds.data import ColumnType, Table, TableSchema


def test_table_equals_valid():
    table = Table.from_json("tests/resources/test_schema_table.json")
    schema_expected = TableSchema(
        ["A", "B"],
        [
            ColumnType.from_numpy_dtype(np.dtype("int64")),
            ColumnType.from_numpy_dtype(np.dtype("int64")),
        ],
    )

    assert table.schema == schema_expected


def test_table_equals_invalid():
    table = Table.from_json("tests/resources/test_schema_table.json")
    schema_not_expected = TableSchema(
        ["A", "C"],
        [
            ColumnType.from_numpy_dtype(np.dtype("f8")),
            ColumnType.from_numpy_dtype(np.dtype("int64")),
        ],
    )

    assert table.schema != schema_not_expected
