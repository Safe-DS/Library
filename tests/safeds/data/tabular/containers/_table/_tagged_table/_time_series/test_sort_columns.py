from collections.abc import Callable

import pytest
from safeds.data.tabular.containers import Column, TimeSeries


@pytest.mark.parametrize(
    ("query", "col1", "col2", "col3", "col4", "col5"),
    [
        (None, 0, 1, 2, 3, 4),
        (
            lambda col1, col2: (col1.name < col2.name) - (col1.name > col2.name),
            4,
            3,
            2,
            1,
            0,
        ),
    ],
    ids=["no query", "with query"],
)
def test_should_return_sorted_table(
    query: Callable[[Column, Column], int],
    col1: int,
    col2: int,
    col3: int,
    col4: int,
    col5: int,
) -> None:
    columns = [
        Column("col1", ["A", "B", "C", "A", "D"]),
        Column("col2", ["Test1", "Test1", "Test3", "Test1", "Test4"]),
        Column("col3", [1, 2, 3, 4, 5]),
        Column("col4", [2, 3, 1, 4, 6]),
        Column("time", [0, 1, 2, 3, 4]),
    ]
    table1 = TimeSeries(
        {
            "col2": ["Test1", "Test1", "Test3", "Test1", "Test4"],
            "col3": [1, 2, 3, 4, 5],
            "col4": [2, 3, 1, 4, 6],
            "col1": ["A", "B", "C", "A", "D"],
            "time": [0, 1, 2, 3, 4],
        },
        target_name="col1",
        time_name="time",
        feature_names=["col4", "col3"],
    )
    if query is not None:
        table_sorted = table1.sort_columns(query)
    else:
        table_sorted = table1.sort_columns()
    table_sorted_columns = table_sorted.to_columns()
    assert table_sorted.schema == table1.schema
    assert table_sorted_columns[0] == columns[col1]
    assert table_sorted_columns[1] == columns[col2]
    assert table_sorted_columns[2] == columns[col3]
    assert table_sorted_columns[3] == columns[col4]
    assert table_sorted_columns[4] == columns[col5]
    assert table_sorted.features == table1.features
    assert table_sorted.target == table1.target
    assert table_sorted.time == table1.time
