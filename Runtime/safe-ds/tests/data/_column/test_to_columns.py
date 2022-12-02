import numpy as np
import pandas as pd
import pytest
from safe_ds.data import Column, ColumnType, Table


@pytest.mark.parametrize(
    "values, name, column_type, index",
    [([1, 4], "A", "int64", 0), ([2, 5], "B", "int64", 1)],
)
def test_to_columns(values: list[int], name: str, column_type: str, index: int):
    table = Table.from_csv("tests/resources/test_column_table.csv")
    columns_list: list[Column] = table.to_columns()

    column_expected: Column = Column(
        pd.Series(values, name=name),
        name,
        ColumnType.from_numpy_dtype(np.dtype(column_type)),
    )

    assert column_expected == columns_list[index]
