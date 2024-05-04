import pandas as pd
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import ColumnType, Integer, Schema


@pytest.mark.parametrize(
    ("table1", "filter_column", "filter_value", "table2"),
    [
        (
            Table(),
            "col1",
            1,
            Table._from_pandas_dataframe(pd.DataFrame(), Schema({})),
        ),
        (
            Table({"col1": [3, 2, 4], "col2": [1, 2, 4]}),
            "col1",
            1,
            Table._from_pandas_dataframe(pd.DataFrame(), Schema({"col1": Integer(), "col2": Integer()})),
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            "col1",
            1,
            Table({"col1": [1, 1], "col2": [1, 4]}),
        ),
    ],
    ids=[
        "empty table",
        "no matches",
        "matches",
    ],
)
def test_should_keep_only_rows(table1: Table, filter_column: str, filter_value: ColumnType, table2: Table) -> None:
    table1 = table1.keep_only_rows(lambda row: row.get_value(filter_column) == filter_value)
    assert table1.schema == table2.schema
    assert table2 == table1
