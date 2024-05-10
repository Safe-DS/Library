from typing import Any

import pytest
from safeds.data.tabular.containers import Table


@pytest.mark.parametrize(
    ("table1", "filter_column", "filter_value", "table2"),
    [
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
        "no match",
        "matches",
    ],
)
def test_should_remove_rows(table1: Table, filter_column: str, filter_value: Any, table2: Table) -> None:
    table1 = table1.remove_rows(lambda row: row.get_value(filter_column) == filter_value)
    assert table1.schema == table2.schema
    assert table2 == table1
