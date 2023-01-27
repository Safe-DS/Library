import numpy as np
import pandas as pd
from safe_ds.data import ColumnType, Row, TableSchema


def test_get_column_names() -> None:
    row = Row(
        pd.Series(data=[1, 2]),
        TableSchema(
            {
                "col1": ColumnType.from_numpy_dtype(np.dtype(float)),
                "col2": ColumnType.from_numpy_dtype(np.dtype(float)),
            }
        ),
    )
    assert row.get_column_names() == ["col1", "col2"]


def test_get_column_names_empty() -> None:
    row = Row(pd.Series(data=[]), TableSchema({}))
    assert not row.get_column_names()
