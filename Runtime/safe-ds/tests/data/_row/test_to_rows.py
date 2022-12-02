import numpy as np
import pandas as pd
from safe_ds.data import Row, Table, TableSchema


def test_to_rows():
    table = Table.from_json("tests/resources/test_row_table.json")
    expected_schema: TableSchema = TableSchema(
        ["A", "B"], [np.dtype("int64"), np.dtype("int64")]
    )
    rows_expected: list[Row] = [
        Row(pd.Series([1, 2], index=["A", "B"], name=0), expected_schema)
    ]

    rows_is: list[Row] = table.to_rows()

    for row_is, row_expected in zip(rows_is, rows_expected):
        assert row_is._data.equals(row_expected._data)
