import pandas as pd
import pytest
from _pytest.python_api import raises
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.typing import ColumnType, Int, String
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError


@pytest.mark.parametrize(
    "column, col_type",
    [
        (Column("col3", ["a", "b", "c"]), String()),
        (Column("col3", [0, -1, -2]), Int()),
    ],
)
def test_add_column_valid(column: Column, col_type: ColumnType) -> None:
    table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    table1 = table1.add_column(column)
    assert table1.count_columns() == 3
    assert table1.get_column("col3") == column
    assert isinstance(table1.schema.get_type_of_column("col1"), Int)
    assert isinstance(table1.schema.get_type_of_column("col2"), Int)
    assert isinstance(table1.schema.get_type_of_column("col3"), type(col_type))


def test_add_column_invalid_duplicate_column_name_error() -> None:
    with raises(DuplicateColumnNameError):
        table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
        table1 = table1.add_column(Column("col1", ["a", "b", "c"]))


def test_add_column_invalid_column_size_error() -> None:
    with raises(ColumnSizeError):
        table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
        table1 = table1.add_column(Column("col3", ["a", "b", "c", "d"]))
