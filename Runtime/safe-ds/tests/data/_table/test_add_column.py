import pandas as pd
import pytest
from _pytest.python_api import raises
from safeds.data import Column, ColumnType, IntColumnType, StringColumnType, Table
from safeds.exceptions import ColumnSizeError, DuplicateColumnNameError


@pytest.mark.parametrize(
    "column, col_type",
    [
        (Column(["a", "b", "c"], "col3"), StringColumnType()),
        (Column([0, -1, -2], "col3"), IntColumnType()),
    ],
)
def test_add_column_valid(column: Column, col_type: ColumnType) -> None:
    table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
    table1 = table1.add_column(column)
    assert table1.count_columns() == 3
    assert table1.get_column("col3") == column
    assert isinstance(table1.schema.get_type_of_column("col1"), IntColumnType)
    assert isinstance(table1.schema.get_type_of_column("col2"), IntColumnType)
    assert isinstance(table1.schema.get_type_of_column("col3"), type(col_type))


def test_add_column_invalid_duplicate_column_name_error() -> None:
    with raises(DuplicateColumnNameError):
        table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
        table1 = table1.add_column(Column(["a", "b", "c"], "col1"))


def test_add_column_invalid_column_size_error() -> None:
    with raises(ColumnSizeError):
        table1 = Table(pd.DataFrame(data={"col1": [1, 2, 1], "col2": [1, 2, 4]}))
        table1 = table1.add_column(Column(["a", "b", "c", "d"], "col3"))
