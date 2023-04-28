import pytest

from safeds.data.tabular.containers import Column, Table
from safeds.data.tabular.typing import Integer, String, ColumnType


@pytest.mark.parametrize(
    ("table1", "column1", "column2", "table2"),
    [
        (Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Column("col3", [0, -1, -2]), Column("col4", ["a", "b", "c"]),
         Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": [0, -1, -2], "col4": ["a", "b", "c"]})),
    ],
    ids=["Integer-String"],
)
# hier soll table1, columns die hinzugefügt werden sollen und table2 (fertiger table) sein, in funktion vergleichen
def test_should_add_columns(table1: Table, column1: Column,
                            column2: Column, table2: Table) -> None:
    table1 = table1.add_columns([column1, column2])
    assert table1 == table2

@pytest.mark.parametrize(
    ("table1", "table2", "expected"),
    [
        (Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4]}),
         Table.from_dict({"col3": [0, -1, -2], "col4": ["a", "b", "c"]}),
         Table.from_dict({"col1": [1, 2, 1], "col2": [1, 2, 4], "col3": [0, -1, -2], "col4": ["a", "b", "c"]})),
    ],
    ids=["Integer-String"],
)
def test_should_add_columns_of_table(table1: Table, table2: Table, expected: Table) -> None:
   table1 = table1.add_columns(table2)
   assert table1 == expected
