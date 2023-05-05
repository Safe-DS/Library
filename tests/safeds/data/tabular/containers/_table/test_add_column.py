import pytest
from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.exceptions import ColumnSizeError, DuplicateColumnNameError
from safeds.data.tabular.typing import ColumnType, Integer, String


@pytest.mark.parametrize(
    ("table1", "column", "col_type", "expected"),
    [
        (Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Column("col3", ["a", "b", "c"]),
         String(),
         Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": ["a", "b", "c"]})),
        (Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Column("col3", [0, -1, -2]),
         Integer(),
         Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": [0, -1, -2]})),
    ],
    ids=["String", "Integer"],
)
def test_should_add_column(table1: Table, column: Column, col_type: ColumnType, expected: Table) -> None:
    table1 = table1.add_column(column)
    assert table1 == expected


def test_should_raise_duplicate_column_name_error() -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    with pytest.raises(DuplicateColumnNameError):
        table1 = table1.add_column(Column("col1", ["a", "b", "c"]))


def test_should_raise_column_size_error() -> None:
    table1 = Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]})
    with pytest.raises(ColumnSizeError):
        table1 = table1.add_column(Column("col3", ["a", "b", "c", "d"]))
