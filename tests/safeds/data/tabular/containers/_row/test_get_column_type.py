import pytest
from safeds.data.tabular.containers import Table
from safeds.data.tabular.containers._lazy_vectorized_row import _LazyVectorizedRow
from safeds.data.tabular.typing import ColumnType


@pytest.mark.parametrize(
    ("table", "column_name", "expected"),
    [
        (Table({"col1": ["A"]}), "col1", "String"),
        (Table({"col1": ["a"], "col2": [1]}), "col2", "Int64"),
    ],
    ids=[
        "one column",
        "two columns",
    ],
)
def test_should_return_the_type_of_the_column(table: Table, column_name: str, expected: ColumnType) -> None:
    row = _LazyVectorizedRow(table=table)
    assert str(row.get_column_type(column_name)) == expected
