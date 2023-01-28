from typing import Callable

import pandas as pd
import pytest
from safeds.data import Column, Table


@pytest.mark.parametrize(
    "query, col1, col2, col3, col4",
    [
        (None, 0, 1, 2, 3),
        (
            lambda col1, col2: (col1.name < col2.name) - (col1.name > col2.name),
            3,
            2,
            1,
            0,
        ),
    ],
)
def test_sort_columns_valid(
    query: Callable[[Column, Column], int], col1: int, col2: int, col3: int, col4: int
) -> None:
    columns = [
        Column(pd.Series(data=["A", "B", "C", "A", "D"]), "col1"),
        Column(pd.Series(data=["Test1", "Test1", "Test3", "Test1", "Test4"]), "col2"),
        Column(pd.Series(data=[1, 2, 3, 4, 5]), "col3"),
        Column(pd.Series(data=[2, 3, 1, 4, 6]), "col4"),
    ]
    table1 = Table(
        pd.DataFrame(
            data={
                "col2": ["Test1", "Test1", "Test3", "Test1", "Test4"],
                "col3": [1, 2, 3, 4, 5],
                "col4": [2, 3, 1, 4, 6],
                "col1": ["A", "B", "C", "A", "D"],
            }
        )
    )
    if query is not None:
        table_sorted = table1.sort_columns(query)
    else:
        table_sorted = table1.sort_columns()
    table_sorted_columns = table_sorted.to_columns()
    assert table_sorted_columns[0] == columns[col1]
    assert table_sorted_columns[1] == columns[col2]
    assert table_sorted_columns[2] == columns[col3]
    assert table_sorted_columns[3] == columns[col4]
    assert table_sorted.count_columns() == 4
    assert table_sorted.count_rows() == table1.count_rows()
