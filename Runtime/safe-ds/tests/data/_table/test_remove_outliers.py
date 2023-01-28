import numpy as np
import pandas as pd
from safeds.data import ColumnType, Table, TableSchema


def test_remove_outliers_no_outliers() -> None:
    table = Table(
        pd.DataFrame(
            data={
                "col1": ["A", "B", "C"],
                "col2": [1.0, 2.0, 3.0],
                "col3": [2, 3, 1],
            }
        )
    )
    names = table.get_column_names()
    result = table.remove_outliers()
    assert result.count_rows() == 3
    assert result.count_columns() == 3
    assert names == table.get_column_names()


def test_remove_outliers_with_outliers() -> None:
    table = Table(
        pd.DataFrame(
            data={
                "col1": [
                    "A",
                    "B",
                    "C",
                    "outlier",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                ],
                "col2": [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "col3": [2, 3, 1, 1_000_000_000, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
    )
    result = table.remove_outliers()
    assert result.count_rows() == 11
    assert result.count_columns() == 3


def test_remove_outliers_no_rows() -> None:
    table = Table(
        [], TableSchema({"col1": ColumnType.from_numpy_dtype(np.dtype(float))})
    )
    result = table.remove_outliers()
    assert result.count_rows() == 0
    assert result.count_columns() == 1
