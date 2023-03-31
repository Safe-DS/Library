import numpy as np
import pandas as pd
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import ColumnType, TableSchema


def test_remove_rows_with_missing_values_valid() -> None:
    table = Table(
        pd.DataFrame(
            data={
                "col1": [None, None, "C", "A"],
                "col2": [None, "Test1", "Test3", "Test1"],
                "col3": [None, 2, 3, 4],
                "col4": [None, 3, 1, 4],
            }
        )
    )
    updated_table = table.remove_rows_with_missing_values()
    assert updated_table.count_rows() == 2


def test_remove_rows_with_missing_values_empty() -> None:
    table = Table([], TableSchema({"col1": ColumnType._from_numpy_dtype(np.dtype(float))}))
    updated_table = table.remove_rows_with_missing_values()
    assert updated_table.get_column_names() == ["col1"]
