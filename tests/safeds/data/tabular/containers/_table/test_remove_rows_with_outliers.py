import pandas as pd

from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import TableSchema, Number


def test_remove_rows_with_outliers_no_outliers() -> None:
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
    result = table.remove_rows_with_outliers()
    assert result.count_rows() == 3
    assert result.count_columns() == 3
    assert names == table.get_column_names()


def test_remove_rows_with_outliers_with_outliers() -> None:
    input_ = Table(
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
                "col2": [1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
                "col3": [2, 3, 1, 1_000_000_000, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
    )
    result = input_.remove_rows_with_outliers()

    expected = Table(
        pd.DataFrame(
            data={
                "col1": ["A", "B", "C", "a", "a", "a", "a", "a", "a", "a", "a"],
                "col2": [1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
                "col3": [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
    )

    assert result == expected


def test_remove_rows_with_outliers_no_rows() -> None:
    table = Table([], TableSchema({"col1": Number()}))
    result = table.remove_rows_with_outliers()
    assert result.count_rows() == 0
    assert result.count_columns() == 1
