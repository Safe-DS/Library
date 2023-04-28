import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import ColumnSizeError, DuplicateColumnNameError
from safeds.data.tabular.typing import ColumnType, Integer, String


@pytest.mark.parametrize(
    ("column", "col_type"),
    [
        (Column("col3", ["a", "b", "c"]), String()),
        (Column("col3", [0, -1, -2]), Integer()),
    ],
    ids=["String", "Integer"],
)
def test_should_add_column(column: Column, col_type: ColumnType) -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    table1 = table1.add_column(column)
    assert table1.number_of_columns == 3
    assert table1.get_column("col3") == column
    assert isinstance(table1.schema.get_column_type("col1"), Integer)
    assert isinstance(table1.schema.get_column_type("col2"), Integer)
    assert isinstance(table1.schema.get_column_type("col3"), type(col_type))


def test_should_raise_DuplicateColumnNameError() -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    with pytest.raises(DuplicateColumnNameError):
        table1 = table1.add_column(Column("col1", ["a", "b", "c"]))


def test_should_raise_ColumnSizeError() -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    with pytest.raises(ColumnSizeError):
        table1 = table1.add_column(Column("col3", ["a", "b", "c", "d"]))
