import re

import pytest

from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow
from safeds.exceptions import ColumnNotFoundError
from tests.helpers import assert_row_operation_works


@pytest.mark.parametrize(
    ("table_data", "column_name", "target", "expected"),
    [
        ({"A": [1, 2]}, "A", 1, {"A": [2]}),
        ({"A": [1, 2, 3], "B": [4, 5, 2]}, "B", 2, {"A": [1, 2], "B": [4, 5]}),
    ],
    ids=[
        "table one column",
        "table two columns",
    ],
)
def test_should_get_correct_item(table_data: dict, column_name: str, target: int, expected: dict) -> None:
    assert_row_operation_works(
        table_data,
        lambda table: table.remove_rows(lambda row: row[column_name].eq(target)),
        expected,
    )


@pytest.mark.parametrize(
    ("table", "column_name"),
    [
        (Table(), "A"),
        (Table({"A": ["a", "aa", "aaa"]}), "B"),
        (Table({"A": ["b", "aa", "aaa"], "C": ["b", "aa", "aaa"]}), "B"),
    ],
    ids=[
        "empty table",
        "table with one column",
        "table with two columns",
    ],
)
def test_should_raise_column_not_found_error(table: Table, column_name: str) -> None:
    row = _LazyVectorizedRow(table=table)
    with pytest.raises(ColumnNotFoundError, match=re.escape(f"Could not find column(s):\n    - '{column_name}'")):
        row[column_name]
