import numpy as np
import pandas as pd
from safe_ds.data import ColumnType, Table, TableSchema


def test_list_columns_with_numerical_values_valid() -> None:
    table = Table(
        pd.DataFrame(
            data={
                "col1": ["A", "B", "C", "A"],
                "col2": ["Test1", "Test1", "Test3", "Test1"],
                "col3": [1, 2, 3, 4],
                "col4": [2, 3, 1, 4],
            }
        )
    )
    columns = table.list_columns_with_numerical_values()
    assert columns[0] == table.get_column("col3")
    assert columns[1] == table.get_column("col4")
    assert len(columns) == 2


def test_list_columns_with_numerical_values_invalid() -> None:
    table = Table(
        [], TableSchema({"col1": ColumnType.from_numpy_dtype(np.dtype(float))})
    )
    table._data["col1"] = None
    columns = table.list_columns_with_numerical_values()
    assert columns[0] == table.get_column("col1")
    assert len(columns) == 1
