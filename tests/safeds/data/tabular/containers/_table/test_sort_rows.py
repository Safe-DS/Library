from typing import Callable

import pytest
from safeds.data.tabular.containers import Column, Row, Table


@pytest.mark.parametrize(
    ("table", "sorter", "expected"),
    [
        # Activate when https://github.com/Safe-DS/Stdlib/issues/75 is fixed.
        # (
        #     Table.from_columns([Column([], "col1")]),
        #     lambda row1, row2: row1["col1"] - row2["col1"],
        #     Table.from_columns([Column([], "col1")])
        # ),
        (
            Table.from_columns([Column([3, 2, 1], "col1")]),
            lambda row1, row2: row1["col1"] - row2["col1"],
            Table.from_columns([Column([1, 2, 3], "col1")]),
        ),
    ],
)
def test_sort_rows(
    table: Table, sorter: Callable[[Row, Row], int], expected: Table
) -> None:
    assert table.sort_rows(sorter) == expected
