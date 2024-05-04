import pandas as pd
import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.typing import ColumnType, Schema


@pytest.mark.parametrize(
    ("table1", "remove_column", "remove_value", "table2"),
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
            Table({"col1": [3, 2, 4], "col2": [1, 2, 4]}),
        ),
        (
            Table({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
            "col1",
            1,
            Table({"col1": [2], "col2": [2]}),
        ),
    ],
    ids=[
        "empty table",
        "no match",
        "matches",
    ],
)
def test_should_remove_rows(table1: Table, remove_column: str, remove_value: ColumnType, table2: Table) -> None:
    table1 = table1.remove_rows(lambda row: row.get_value(remove_column) == remove_value)
    assert table1.schema == table2.schema
    assert table2 == table1
